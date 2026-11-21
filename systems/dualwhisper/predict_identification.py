"""Identification of given vocalizations with DualWhisper.

    python -m systems.dualwhisper.predict_identification --data DIR --size 50

For every example of dataset/classification/test, one model window is cut from the 5-second
example, starting two time tokens before the given onset, so that the given vocalization is the
first one of the window. The decoder receives the training prefix followed by the time token
of the given onset; the taxon it then generates is the prediction. Of five beam hypotheses,
the most probable one that names a taxon is used; without such a hypothesis the prediction is
"unknown". Writes predictions/dualwhisper/identification_<size>.json.
"""
import argparse
import io
import json
import re
import sys

import ctranslate2
import numpy as np
import soundfile as sf

from benchmark import config, data
from systems.whisperseg import common as whisperseg

sys.path.insert(0, str(whisperseg.WHISPERSEG))
from audio_utils import WhisperSegFeatureExtractor  # noqa: E402
from model import WhisperSegmenterFast  # noqa: E402

# One time token of the decoder spans this many spectrogram columns.
COLUMNS_PER_TIME_TOKEN = 2
WINDOW_COLUMNS = 1000
LEAD_TIME_TOKENS = 2
BATCH_SIZE = 16
BEAM_SIZE = 5
PREFIX_TOKENS = ["<|startoftranscript|>", "<|en|>", "<|notimestamps|>"]


def corpus_settings():
    """Return corpus -> (spectrogram time step, minimum frequency) of the prepared test set."""
    settings = {}
    for path in sorted(config.prepared_dir("test").glob("*.json")):
        corpus = data.corpus_of(path.name)
        if corpus not in settings:
            labels = json.loads(path.read_text())
            settings[corpus] = (labels["spec_time_step"], labels["min_frequency"])
    return settings


def window_features(row, time_step, min_frequency, extractors):
    """Return (log-mel features of the model window, index of the onset time token)."""
    waveform, sampling_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(1)
    window_seconds = WINDOW_COLUMNS * time_step
    onset = float(row["onset_in_window"])
    start = max(0.0, onset - LEAD_TIME_TOKENS * time_step * COLUMNS_PER_TIME_TOKEN)
    first_sample = int(start * sampling_rate)
    window_samples = int(window_seconds * sampling_rate)
    window = waveform[first_sample:first_sample + window_samples]
    padding = np.zeros(window_samples - len(window), np.float32)
    window = np.concatenate([window, padding])

    key = (sampling_rate, time_step, min_frequency)
    if key not in extractors:
        extractors[key] = WhisperSegFeatureExtractor(
            sampling_rate,
            time_step,
            min_frequency=min_frequency,
            chunk_length=max(30, int(np.ceil(window_seconds))),
        )
    extracted = extractors[key](window, sampling_rate=sampling_rate, padding="do_not_pad")
    features = extracted["input_features"][0][:, :WINDOW_COLUMNS]
    if features.shape[1] < WINDOW_COLUMNS:
        missing = WINDOW_COLUMNS - features.shape[1]
        floor = features.min() * np.ones((features.shape[0], missing))
        features = np.concatenate([features, floor], 1)
    onset_token = int(np.round((onset - start) / (time_step * COLUMNS_PER_TIME_TOKEN)))
    onset_token = min(onset_token, WINDOW_COLUMNS)
    return features.astype(np.float32), onset_token


def taxon_of_hypothesis(tokens, onset_token, taxon_of_label_id):
    """Return the taxon named after the onset token in one beam hypothesis, or ''.

    The hypothesis may or may not repeat the prompt. The model writes the label as its number in
    the label table of the checkpoint."""
    text = "".join(tokens).replace("Ġ", " ")
    match = re.search(r"<\|%d\|>([^<]+?)<\|\d+\|>" % onset_token, text)
    if match is None:
        match = re.match(r"([^<]+?)<\|\d+\|>", text)
    if match is None:
        return ""
    label = match.group(1).strip()
    if label.isdigit():
        return taxon_of_label_id.get(int(label), "unknown")
    return label


def decode_batch(model, features, prompts, onset_tokens, max_length, taxon_of_label_id):
    """Return the predicted taxon of every window of one batch."""
    results = model.generate(
        ctranslate2.StorageView.from_array(np.asarray(features)),
        prompts,
        max_length=max_length,
        beam_size=BEAM_SIZE,
        num_hypotheses=BEAM_SIZE,
        suppress_blank=False,
    )
    taxa = []
    for result, onset_token in zip(results, onset_tokens):
        predicted = "unknown"
        for tokens in result.sequences:
            taxon = taxon_of_hypothesis(tokens, onset_token, taxon_of_label_id)
            if taxon and taxon != "unknown":
                predicted = taxon
                break
        taxa.append(predicted)
    return taxa


def identify(rows, size):
    """Return {example id: predicted taxon} for identification examples `rows`."""
    model_dir = config.DATA / "models" / "dualwhisper" / str(size)
    segmenter = WhisperSegmenterFast(str(model_dir / "final_checkpoint_ct2"), device="cuda")
    model = segmenter.model_list[0]
    tokenizer = segmenter.tokenizer_list[0]
    taxon_of_label_id = segmenter.inverse_cluster_codebook
    # The prefix of every training target: the start tokens and the species token as plain text.
    prefix = tokenizer.convert_tokens_to_ids(PREFIX_TOKENS + tokenizer.tokenize("<|unknown|>"))
    max_length = len(prefix) + 32

    settings = corpus_settings()
    extractors = {}
    predicted = {}
    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start:batch_start + BATCH_SIZE]
        features = []
        prompts = []
        onset_tokens = []
        for row in batch:
            time_step, min_frequency = settings[row["dataset"]]
            window, onset_token = window_features(row, time_step, min_frequency, extractors)
            features.append(window)
            prompts.append(prefix + [tokenizer.convert_tokens_to_ids("<|%d|>" % onset_token)])
            onset_tokens.append(onset_token)
        taxa = decode_batch(
            model, features, prompts, onset_tokens, max_length, taxon_of_label_id
        )
        for row, taxon in zip(batch, taxa):
            predicted[data.example_id(row)] = taxon
        if batch_start % (100 * BATCH_SIZE) == 0:
            print(f"{batch_start}/{len(rows)} examples", flush=True)
    return predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: environment variable CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    rows = data.table(
        "classification", "test", ("chunk_id", "event_index", "audio", "onset_in_window")
    )
    predicted = identify(rows, arguments.size)
    output = config.prediction_file("dualwhisper", "identification", arguments.size)
    data.write_json(output, predicted)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
