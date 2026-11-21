"""Identification examples and the audio given to encoders that take one clip per example."""
import functools
import io

import numpy as np
import pyarrow.compute
import pyarrow.parquet
import soundfile

from benchmark import config, data

COLUMNS = (
    "chunk_id",
    "event_index",
    "species",
    "rec_id",
    "sampling_rate",
    "onset_in_window",
    "offset_in_window",
)


@functools.lru_cache(maxsize=None)
def encoded_audio(split):
    """Return the audio column of one split, in the row order of data.table()."""
    files = sorted((config.dataset_dir() / "classification").glob(f"{split}-*.parquet"))
    file_names = [str(file) for file in files]
    table = pyarrow.parquet.ParquetDataset(file_names).read(columns=["audio", "dataset"])
    kept = pyarrow.compute.not_equal(table["dataset"], config.DUPLICATE_CORPUS)
    return table.filter(kept)["audio"]


def examples(split):
    """Return the identification examples of one split as dictionaries, in dataset order.

    split: 'train_50', 'train_100', 'train_500' or 'test'. 'split' and 'row' locate the audio
    of an example for waveform()."""
    rows = []
    for index, row in enumerate(data.table("classification", split, COLUMNS)):
        rows.append(dict(row, split=split, row=index))
    return rows


def embedding_examples():
    """Return the examples in the row order of an embedding file: train_500, then test."""
    return examples("train_500") + examples("test")


def waveform(example):
    """Return the 5-s window of an example at its own sampling rate, channels averaged."""
    encoded = encoded_audio(example["split"])[example["row"]].as_py()["bytes"]
    samples, _ = soundfile.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
    return samples.mean(axis=1)


def resampled(samples, sampling_rate, target_rate):
    """Return the samples at `target_rate`, by linear interpolation."""
    if target_rate == sampling_rate:
        return samples
    length = int(round(len(samples) * target_rate / sampling_rate))
    positions = np.linspace(0, len(samples) - 1, length, dtype=np.float64)
    source_positions = np.arange(len(samples), dtype=np.float64)
    return np.interp(positions, source_positions, samples).astype(np.float32)


def vocalization(example, target_rate, minimum_seconds=0.0):
    """Return the audio between onset and offset of an example, resampled to `target_rate`.

    A vocalization shorter than `minimum_seconds` is widened on both sides with the surrounding
    audio of its window."""
    samples = waveform(example)
    sampling_rate = example["sampling_rate"]
    first = max(0, int(round(example["onset_in_window"] * sampling_rate)))
    end = min(len(samples), int(round(example["offset_in_window"] * sampling_rate)))
    minimum_length = int(round(minimum_seconds * sampling_rate))
    if end - first < minimum_length:
        centre = (first + end) // 2
        first = max(0, centre - minimum_length // 2)
        end = min(len(samples), first + minimum_length)
        first = max(0, end - minimum_length)
    if end <= first:
        return np.zeros(1, dtype=np.float32)
    return resampled(samples[first:end], sampling_rate, target_rate)
