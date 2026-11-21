"""Detection+Identification cascade: WhisperSeg detects, WhisperID names each detection.

    python -m systems.cascade.predict --data DIR --size 50

Reads the WhisperSeg detections predictions/whisperseg/detection_<size>.json.gz for the chunks
of labels/end_to_end_chunks.json and the fine-tuned WhisperID model
models/whisperid/finetuned_<size>/. Every detected vocalization is shown to WhisperID in a
5-second window centred on it, with the detected span marked. The score of a taxon is the
probability that the model generates its name, averaged over three circular time shifts of the
window and divided by the frequency of the taxon in the training examples raised to the power
1.6. The best taxon is written with the detected boundaries to
predictions/cascade/end_to_end_<size>.json.gz.
"""
import argparse
import os
import random
import sys
from collections import Counter, defaultdict

# Set before PyTorch and Transformers are loaded: the PyTorch compiler and TensorFlow are not
# needed, and loading both can crash the process.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from benchmark import config, data  # noqa: E402
from systems.whisperseg import common as whisperseg  # noqa: E402

sys.path.insert(0, str(whisperseg.WHISPERSEG))
from systems.whisperid import features, model as whisperid  # noqa: E402

# Circular shifts of the window audio in seconds; the marker moves with the audio.
SHIFTS = (-0.25, 0.0, 0.25)
# The shift probabilities are added in this order (indices of SHIFTS).
AVERAGING_ORDER = (1, 0, 2)
# Exponent of the training frequency that is divided out of the taxon probabilities.
PRIOR_EXPONENT = 1.6
# Number of windows per forward pass.
BATCH_SIZE = 12
# Validation split of the WhisperID fine-tuning.
VALIDATION_FRACTION = 0.1
VALIDATION_SEED = 42


def whisperid_training_examples(size):
    """Return the identification examples WhisperID was fine-tuned on at one training size.

    These are the examples of dataset/classification/train_<size> without the validation
    examples: for every taxon with at least two recordings, a tenth of its recordings (at least
    one, never all), drawn with a fixed seed, is held out with all their examples."""
    rows = data.table("classification", f"train_{size}", ("species", "rec_id"))
    by_taxon = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_taxon[row["species"]][row["rec_id"]].append(row)
    generator = random.Random(VALIDATION_SEED)
    training = []
    for by_recording in by_taxon.values():
        recordings = sorted(by_recording)
        generator.shuffle(recordings)
        held_out = 0
        if len(recordings) >= 2:
            held_out = int(round(VALIDATION_FRACTION * len(recordings)))
            held_out = min(max(1, held_out), len(recordings) - 1)
        for recording in recordings[held_out:]:
            training.extend(by_recording[recording])
    return training


def training_taxa(size):
    """Return (sorted taxa, number of training examples of each) of the WhisperID training set."""
    counts = Counter()
    for example in whisperid_training_examples(size):
        counts[example["species"]] += 1
    taxa = sorted(counts)
    return taxa, np.array([counts[taxon] for taxon in taxa], np.float64)


def shifted_windows(chunk_audio, sampling_rate, onset, offset):
    """Return (marked spectrograms for SHIFTS, validity of each shift) of one detection.

    The window is centred on the detection and moved to lie inside the chunk. A shift other
    than zero is valid only if the shifted detection stays inside the window."""
    window_samples = int(features.WINDOW_SECONDS * sampling_rate)
    chunk_seconds = len(chunk_audio) / sampling_rate
    start = max(0.0, 0.5 * (onset + offset) - features.WINDOW_SECONDS / 2)
    start = min(start, max(0.0, chunk_seconds - features.WINDOW_SECONDS))
    first_sample = int(start * sampling_rate)
    window = chunk_audio[first_sample:first_sample + window_samples]
    window_seconds = min(features.WINDOW_SECONDS, len(window) / sampling_rate)
    onset_in_window = onset - start
    offset_in_window = offset - start

    spectrograms = []
    valid = []
    for shift in SHIFTS:
        inside = onset_in_window + shift >= 0 and offset_in_window + shift <= window_seconds
        if shift != 0.0 and not inside:
            spectrograms.append(np.zeros((80, features.COLUMNS), np.float32))
            valid.append(False)
            continue
        shifted = window
        applied_shift = 0.0
        if shift and len(window) > 8:
            shift_samples = int(round(shift * sampling_rate))
            shifted = np.roll(window, shift_samples)
            applied_shift = shift_samples / float(sampling_rate)
        if len(shifted) < window_samples:
            silence = np.zeros(window_samples - len(shifted), "float32")
            shifted = np.concatenate([shifted, silence])
        bands = features.spectrogram(shifted, sampling_rate)
        bands = features.mark(
            bands, onset_in_window + applied_shift, offset_in_window + applied_shift
        )
        spectrograms.append(bands)
        valid.append(True)
    return np.stack(spectrograms), np.array(valid)


class TaxonScorer:
    """Log-probability of every taxon name as continuation of the prompt, in one decoder pass.

    The token sequences of all names (each ending with the end-of-text token) form a prefix
    tree. Every tree node is one decoder position that attends to the prompt and to its own
    ancestors, so that the decoder input is the prompt followed by all nodes."""

    def __init__(self, model, tokenizer, taxa):
        self.model = model
        self.device = model.device
        end_of_text = tokenizer.eos_token_id
        names = []
        for taxon in taxa:
            names.append(tokenizer.encode(" " + taxon, add_special_tokens=False) + [end_of_text])

        # A node stands for a non-empty proper prefix of a name: (last token, parent, depth).
        self.nodes = []
        node_of_prefix = {}
        for name in names:
            for length in range(1, len(name)):
                prefix = tuple(name[:length])
                if prefix not in node_of_prefix:
                    node_of_prefix[prefix] = len(self.nodes)
                    parent = node_of_prefix.get(tuple(name[:length - 1]), -1)
                    self.nodes.append((name[length - 1], parent, length))
        node_count = len(self.nodes)
        self.node_tokens = torch.tensor([node[0] for node in self.nodes], device=self.device)
        self.ancestors = torch.zeros(node_count, node_count, dtype=torch.bool)
        for index, (_, parent, _) in enumerate(self.nodes):
            self.ancestors[index, index] = True
            if parent >= 0:
                self.ancestors[index] |= self.ancestors[parent]

        # The first token of a name is predicted at the last prompt position; token k + 1 is
        # predicted at the node of the first k tokens.
        step_nodes = []
        step_targets = []
        step_taxa = []
        for taxon_index, name in enumerate(names):
            for length in range(1, len(name)):
                step_nodes.append(node_of_prefix[tuple(name[:length])])
                step_targets.append(name[length])
                step_taxa.append(taxon_index)
        self.first_tokens = torch.tensor([name[0] for name in names], device=self.device)
        self.step_nodes = torch.tensor(step_nodes, device=self.device)
        self.step_targets = torch.tensor(step_targets, device=self.device)
        self.step_taxa = torch.tensor(step_taxa, device=self.device)
        self.masks = {}

    def mask_and_positions(self, prompt_length):
        """Return (additive attention mask, position of every decoder input)."""
        if prompt_length not in self.masks:
            node_count = len(self.nodes)
            total = prompt_length + node_count
            allowed = torch.zeros(total, total, dtype=torch.bool)
            causal = torch.tril(torch.ones(prompt_length, prompt_length, dtype=torch.bool))
            allowed[:prompt_length, :prompt_length] = causal
            allowed[prompt_length:, :prompt_length] = True
            allowed[prompt_length:, prompt_length:] = self.ancestors
            mask = torch.zeros(total, total, dtype=torch.float16)
            mask[~allowed] = torch.finfo(torch.float16).min
            node_positions = [prompt_length + depth - 1 for _, _, depth in self.nodes]
            positions = torch.cat([torch.arange(prompt_length), torch.tensor(node_positions)])
            self.masks[prompt_length] = (mask.to(self.device), positions.to(self.device))
        return self.masks[prompt_length]

    def scores(self, encoded, prompt):
        """Return the (windows, taxa) log-probabilities for encoder outputs `encoded`."""
        batch = encoded.shape[0]
        prompt_length = len(prompt)
        mask, positions = self.mask_and_positions(prompt_length)
        tokens = torch.cat([torch.tensor(prompt, device=self.device), self.node_tokens])
        decoded = self.model.model.decoder(
            input_ids=tokens[None].expand(batch, -1),
            encoder_hidden_states=encoded,
            attention_mask=mask[None, None].expand(batch, 1, -1, -1),
            position_ids=positions[None].expand(batch, -1),
            use_cache=False,
        )
        hidden = decoded.last_hidden_state[:, prompt_length - 1:]
        log_probabilities = torch.log_softmax(self.model.proj_out(hidden).float(), -1)
        totals = log_probabilities[:, 0, self.first_tokens].clone()
        steps = log_probabilities[:, 1 + self.step_nodes, self.step_targets]
        totals.index_add_(1, self.step_taxa, steps)
        return totals.cpu().numpy()


def best_taxon(shift_scores, valid, training_counts):
    """Return the index of the best taxon of one detection.

    shift_scores: (shifts, taxa) log-probabilities; rows of invalid shifts are ignored."""
    summed = np.zeros(shift_scores.shape[1], np.float32)
    for shift_index in AVERAGING_ORDER:
        if valid[shift_index]:
            scores = shift_scores[shift_index]
            probabilities = np.exp(scores - scores.max())
            summed = summed + probabilities / probabilities.sum()
    mean = summed / valid.sum()
    adjusted = np.log(mean + 1e-30) - PRIOR_EXPONENT * np.log(training_counts)
    return int(adjusted.argmax())


@torch.no_grad()
def name_detections(model, scorer, prompt, chunk_audio, sampling_rate, detections, counts):
    """Return the index of the predicted taxon of every detection of one chunk."""
    spectrograms = []
    valid = []
    for onset, offset in detections:
        windows, window_valid = shifted_windows(chunk_audio, sampling_rate, onset, offset)
        spectrograms.append(windows)
        valid.append(window_valid)
    inputs = []
    for detection_index in range(len(detections)):
        for shift_index in range(len(SHIFTS)):
            if valid[detection_index][shift_index]:
                inputs.append((detection_index, shift_index))

    scores = np.full((len(detections), len(SHIFTS), len(counts)), np.nan, np.float32)
    for batch_start in range(0, len(inputs), BATCH_SIZE):
        batch = inputs[batch_start:batch_start + BATCH_SIZE]
        stacked = np.stack([spectrograms[detection][shift] for detection, shift in batch])
        encoder_input = torch.from_numpy(stacked).to(model.device, torch.float16)
        encoded = model.model.encoder(encoder_input).last_hidden_state
        batch_scores = scorer.scores(encoded, prompt)
        for row, (detection, shift) in enumerate(batch):
            scores[detection, shift] = batch_scores[row]

    taxa = []
    for detection_index in range(len(detections)):
        taxa.append(best_taxon(scores[detection_index], valid[detection_index], counts))
    return taxa


def predict(chunk_ids, size):
    """Return the prediction rows of test chunks `chunk_ids` at one training size."""
    taxa, counts = training_taxa(size)
    checkpoint = str(config.DATA / "models" / "whisperid" / f"finetuned_{size}")
    model, tokenizer = whisperid.load(checkpoint, torch.device("cuda"), torch.float16)
    model.eval()
    scorer = TaxonScorer(model, tokenizer, taxa)
    detections = data.predictions("whisperseg", "detection", size)

    rows = []
    for index, chunk_id in enumerate(chunk_ids):
        named = []
        if detections[chunk_id]:
            wav_file = config.prepared_dir("test") / f"{chunk_id}.wav"
            chunk_audio, sampling_rate = sf.read(str(wav_file), dtype="float32")
            if chunk_audio.ndim > 1:
                chunk_audio = chunk_audio.mean(axis=1)
            prompt = whisperid.prompt_tokens(tokenizer, sampling_rate)
            taxon_indices = name_detections(
                model, scorer, prompt, chunk_audio, sampling_rate, detections[chunk_id], counts
            )
            for (onset, offset), taxon_index in zip(detections[chunk_id], taxon_indices):
                named.append([onset, offset, taxa[taxon_index]])
        rows.append({"chunk_id": chunk_id, "pred": named})
        if index % 25 == 0:
            print(f"{index}/{len(chunk_ids)} chunks", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: environment variable CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    chunk_ids = data.read_json(config.label_file("end_to_end_chunks.json"))
    rows = predict(chunk_ids, arguments.size)
    output = config.prediction_file("cascade", "end_to_end", arguments.size)
    data.write_json(output, rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
