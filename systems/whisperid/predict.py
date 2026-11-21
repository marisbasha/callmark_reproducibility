"""Identify the test examples with a fine-tuned WhisperID model.

    python -m systems.whisperid.predict --data DIR --size 50

Reads models/whisperid/finetuned_<size> and writes predictions/whisperid/identification_<size>.json.
Every taxon of the training set is scored by the log-probability of its name; the scores are
averaged over three time shifts of the audio and corrected for the taxon frequencies of the
training set.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from benchmark import config, data
from systems.whisperid import examples, features, model

# Circular shifts of the audio in seconds. A shift is used when the shifted vocalization stays
# inside the audio.
SHIFTS = (-0.25, 0.0, 0.25)
# Weight of the log training frequency that is subtracted from the score of a taxon.
FREQUENCY_CORRECTION = 1.6
WORKERS = 6


def valid_shifts(example, waveform):
    """Return the shifts that keep the vocalization of an example inside its audio."""
    duration = min(features.WINDOW_SECONDS, len(waveform) / example["sampling_rate"])
    shifts = []
    for shift in SHIFTS:
        inside = example["onset_in_window"] + shift >= 0
        inside = inside and example["offset_in_window"] + shift <= duration
        if shift == 0.0 or inside:
            shifts.append(shift)
    return shifts


class ShiftedExamples(Dataset):
    """The spectrograms of every example under its valid shifts."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        example = self.rows[index]
        waveform = examples.waveform(example)
        spectrograms = []
        for shift in valid_shifts(example, waveform):
            spectrograms.append(features.example_features(example, waveform, shift))
        return index, np.stack(spectrograms)


def taxa_and_frequencies(size):
    """Return the sorted taxa of the training examples of one size and their counts."""
    rows = examples.rows(f"train_{size}", with_audio=False)
    training, _ = examples.split_validation(rows)
    counts = {}
    for example in training:
        counts[example["species"]] = counts.get(example["species"], 0) + 1
    taxa = sorted(counts)
    return taxa, np.array([counts[taxon] for taxon in taxa], dtype=np.float64)


def name_tokens(tokenizer, taxa):
    """Return the tokens of every taxon name, each followed by the end token."""
    names = []
    for taxon in taxa:
        tokens = tokenizer.encode(" " + taxon, add_special_tokens=False)
        names.append(tokens + [tokenizer.eos_token_id])
    return names


@torch.no_grad()
def name_scores(whisper, encoded, prompt, names, pad_token):
    """Return the log-probability of every taxon name given one encoded spectrogram."""
    longest = max(len(name) for name in names)
    decoder_input = torch.full((len(names), len(prompt) + longest - 1), pad_token)
    targets = torch.full((len(names), longest), -1)
    for index, name in enumerate(names):
        tokens = prompt + name[:-1]
        decoder_input[index, :len(tokens)] = torch.tensor(tokens)
        targets[index, :len(name)] = torch.tensor(name)
    decoder_input = decoder_input.to(whisper.device)
    targets = targets.to(whisper.device)
    hidden = whisper.model.decoder(
        input_ids=decoder_input,
        encoder_hidden_states=encoded.expand(len(names), -1, -1),
    ).last_hidden_state
    # The output at the last prompt token predicts the first token of the name.
    logits = whisper.proj_out(hidden[:, len(prompt) - 1:]).float()
    log_probabilities = torch.log_softmax(logits, dim=-1)
    of_targets = log_probabilities.gather(2, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (of_targets * (targets >= 0)).sum(1).cpu().numpy()


def corrected_scores(scores_of_shifts, frequencies):
    """Return the scores of the taxa: mean probability over shifts, frequency-corrected."""
    scores = np.stack(scores_of_shifts)
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    mean = probabilities.mean(axis=0)
    return np.log(mean + 1e-30) - FREQUENCY_CORRECTION * np.log(frequencies)


@torch.no_grad()
def identify(whisper, tokenizer, rows, taxa, frequencies):
    """Return the predicted taxon of every example of `rows`."""
    names = name_tokens(tokenizer, taxa)
    predicted = [None] * len(rows)
    loader = DataLoader(ShiftedExamples(rows), batch_size=None, num_workers=WORKERS)
    for count, (index, spectrograms) in enumerate(loader):
        prompt = model.prompt_tokens(tokenizer, rows[index]["sampling_rate"])
        inputs = torch.as_tensor(spectrograms).to(whisper.device, whisper.dtype)
        encoded = whisper.model.encoder(inputs).last_hidden_state
        scores_of_shifts = []
        for position in range(len(encoded)):
            scores_of_shifts.append(name_scores(
                whisper, encoded[position:position + 1], prompt, names, tokenizer.pad_token_id
            ))
        best = corrected_scores(scores_of_shifts, frequencies).argmax()
        predicted[index] = taxa[best]
        if count % 500 == 0:
            print(f"{count}/{len(rows)}", flush=True)
    return predicted


def main():
    """Write the identification predictions of one training size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    args = parser.parse_args()
    config.use_data_root(args.data)

    checkpoint = config.DATA / "models" / "whisperid" / f"finetuned_{args.size}"
    whisper, tokenizer = model.load(str(checkpoint), torch.device("cuda"), torch.float16)
    whisper.eval()
    taxa, frequencies = taxa_and_frequencies(args.size)
    rows = examples.rows("test")
    predicted = identify(whisper, tokenizer, rows, taxa, frequencies)
    ids = [data.example_id(example) for example in rows]
    output_file = config.prediction_file("whisperid", "identification", args.size)
    data.write_json(output_file, dict(zip(ids, predicted)))
    print(f"wrote {output_file}", flush=True)


if __name__ == "__main__":
    main()
