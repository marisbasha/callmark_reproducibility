"""Identification examples of the dataset: rows, audio and the validation split."""
import functools
import io
import random
from collections import defaultdict

import pyarrow.parquet as pq
import soundfile

from benchmark import config

VALIDATION_FRACTION = 0.1
VALIDATION_SEED = 42


def dataset_files(split):
    """Return the names of the dataset files of one split, in dataset order."""
    files = sorted((config.dataset_dir() / "classification").glob(f"{split}-*.parquet"))
    return [str(file) for file in files]


@functools.lru_cache(maxsize=None)
def encoded_audio(split):
    """Return the audio column of one split, read into memory once."""
    return pq.ParquetDataset(dataset_files(split)).read(columns=["audio"]).column("audio")


def rows(split, with_audio=True):
    """Return the examples of one split as dictionaries without audio, in dataset order.

    split: 'train_50', 'train_100', 'train_500' or 'test'. 'table' and 'row' locate the audio
    of an example for waveform(). With `with_audio`, the encoded audio of the split is read
    into memory here, before data-loading processes are started."""
    dataset = pq.ParquetDataset(dataset_files(split))
    columns = [name for name in dataset.schema.names if name != "audio"]
    examples = []
    for index, example in enumerate(dataset.read(columns=columns).to_pylist()):
        if example["dataset"] == config.DUPLICATE_CORPUS:
            continue
        example["table"] = split
        example["row"] = index
        examples.append(example)
    if with_audio:
        encoded_audio(split)
    return examples


def waveform(example):
    """Return the audio of an example at its own sampling rate, channels averaged."""
    encoded = encoded_audio(example["table"])[example["row"]].as_py()["bytes"]
    samples, _ = soundfile.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
    return samples.mean(axis=1)


def split_validation(examples):
    """Return (training examples, validation examples) of a training set.

    For every taxon with at least two recordings, a tenth of its recordings (at least one,
    never all) is held out with all their examples."""
    generator = random.Random(VALIDATION_SEED)
    by_taxon = defaultdict(lambda: defaultdict(list))
    for example in examples:
        by_taxon[example["species"]][example["rec_id"]].append(example)
    training = []
    validation = []
    for by_recording in by_taxon.values():
        recordings = sorted(by_recording)
        generator.shuffle(recordings)
        if len(recordings) < 2:
            held_out = 0
        else:
            held_out = int(round(VALIDATION_FRACTION * len(recordings)))
            held_out = min(max(1, held_out), len(recordings) - 1)
        for recording in recordings[:held_out]:
            validation.extend(by_recording[recording])
        for recording in recordings[held_out:]:
            training.extend(by_recording[recording])
    return training, validation
