"""Detection chunks of the dataset: audio, spectrogram parameters, taxon of every annotation."""
import io
import json
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import soundfile

from benchmark import config, data

# Lowest frequency of the spectrogram, per corpus (Hz). Right whale and spider recordings are
# part of the training material only.
MIN_FREQUENCY = {
    "budgie": 0,
    "canary": 0,
    "enabirds": 0,
    "infant_marmosets": 0,
    "nbm": 0,
    "ozf": 0,
    "right_whale": 0,
    "rodent_usv": 15000,
    "singing_mice": 10000,
    "sne_full": 0,
    "spiders": 0,
    "tworadiobirds": 0,
    "wytham": 0,
}

# Recordings sampled at this rate or above are ultrasonic and get a five times finer spectrogram.
ULTRASONIC_RATE = 200000

# Corpora in which every vocalization belongs to one taxon.
SINGLE_TAXON_CORPORA = (
    "budgie",
    "canary",
    "infant_marmosets",
    "ozf",
    "right_whale",
    "singing_mice",
    "spiders",
    "tworadiobirds",
    "wytham",
)

# Corpora whose recordings hold several taxa: an annotation without an identification example
# stays unidentified.
MIXED_RECORDING_CORPORA = ("enabirds", "nbm", "sne_full")

# An identification example and its annotation agree to this many seconds (onset plus offset).
JOIN_TOLERANCE = 0.002


def spectrogram_parameters(corpus, sample_rate):
    """Return the analysis parameters stored with every prepared file of a corpus."""
    if sample_rate >= ULTRASONIC_RATE:
        time_step = 0.0005
        min_segment_length = 0.005
        eps = 0.005
        scoring_frame = 0.0002
    else:
        time_step = 0.0025
        min_segment_length = 0.010
        eps = 0.020
        scoring_frame = 0.0010
    return {
        "sr": int(sample_rate),
        "min_frequency": min(MIN_FREQUENCY[corpus], sample_rate / 2 - 1),
        "spec_time_step": time_step,
        "min_segment_length": min_segment_length,
        "tolerance": 0.01,
        "time_per_frame_for_scoring": scoring_frame,
        "eps": eps,
    }


def dataset_files(split):
    """Return the files of one split of the detection chunks, in dataset order."""
    return sorted((config.dataset_dir() / "segmentation").glob(f"{split}-*.parquet"))


def rows_with_audio(dataset_file):
    """Yield the rows of one dataset file with their encoded audio, a few rows at a time."""
    for batch in pq.ParquetFile(dataset_file).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            if row["dataset"] != config.DUPLICATE_CORPUS:
                yield row


def decode_audio(row):
    """Return (waveform, sample rate) of a chunk at its own rate, channels averaged."""
    encoded = io.BytesIO(row["audio"]["bytes"])
    waveform, sample_rate = soundfile.read(encoded, dtype="float32", always_2d=True)
    return waveform.mean(axis=1), int(sample_rate)


def audio_length(row):
    """Return the length of a chunk's audio in seconds without decoding it."""
    info = soundfile.info(io.BytesIO(row["audio"]["bytes"]))
    return info.frames / info.samplerate


def write_pair(directory, name, waveform, sample_rate, labels):
    """Write <name>.wav (16-bit) and <name>.json."""
    soundfile.write(directory / f"{name}.wav", waveform, sample_rate, subtype="PCM_16")
    with open(directory / f"{name}.json", "w") as handle:
        json.dump(labels, handle)


def fallback_taxon(chunk, corpus_taxa, recording_taxa):
    """Return the taxon of annotations without an identification example, or None.

    It is the taxon of a single-taxon corpus, otherwise the only taxon among the identification
    examples of the recording, except in corpora whose recordings hold several taxa."""
    corpus = chunk["dataset"]
    if corpus in SINGLE_TAXON_CORPORA and len(corpus_taxa[corpus]) == 1:
        return next(iter(corpus_taxa[corpus]))
    if corpus in MIXED_RECORDING_CORPORA:
        return None
    if len(recording_taxa[chunk["rec_id"]]) == 1:
        return next(iter(recording_taxa[chunk["rec_id"]]))
    return None


def annotation_taxa(split):
    """Return chunk_id -> taxon of every annotation of the chunk (None: unidentified).

    Every identification example of the split names the taxon of its own annotation: the
    annotation of the same chunk with the nearest onset and offset that no earlier example
    took. The remaining annotations get the fallback taxon of the chunk."""
    chunks = data.table(
        "segmentation", split, ("chunk_id", "rec_id", "chunk_start", "onsets", "offsets")
    )
    examples = data.table(
        "classification",
        split,
        ("chunk_id", "rec_id", "species", "onset_in_recording", "offset_in_recording"),
    )
    examples_of_chunk = defaultdict(list)
    corpus_taxa = defaultdict(set)
    recording_taxa = defaultdict(set)
    for example in examples:
        examples_of_chunk[example["chunk_id"]].append(example)
        corpus_taxa[example["dataset"]].add(example["species"])
        recording_taxa[example["rec_id"]].add(example["species"])

    taxa_of_chunk = {}
    for chunk in chunks:
        onsets = np.array(chunk["onsets"], float)
        offsets = np.array(chunk["offsets"], float)
        taxa = [fallback_taxon(chunk, corpus_taxa, recording_taxa)] * len(onsets)
        taken = set()
        for example in examples_of_chunk[chunk["chunk_id"]]:
            onset = example["onset_in_recording"] - chunk["chunk_start"]
            offset = example["offset_in_recording"] - chunk["chunk_start"]
            distance = np.abs(onsets - onset) + np.abs(offsets - offset)
            for index in np.argsort(distance):
                assert distance[index] <= JOIN_TOLERANCE, chunk["chunk_id"]
                if index not in taken:
                    taken.add(index)
                    taxa[index] = example["species"]
                    break
        taxa_of_chunk[chunk["chunk_id"]] = taxa
    return taxa_of_chunk
