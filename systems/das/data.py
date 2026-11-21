"""Data handling shared by the DAS and TweetyNet detectors.

Both detectors are trained as one model for all corpora on the joined training files
(prepared/train_<size>_joined/), with every recording resampled to 32 kHz, and both mark a
vocalization wherever the frame-wise probability of the vocal class reaches 0.5.
"""
import hashlib
import json
from collections import defaultdict

import numpy as np
import soundfile

from benchmark import config, data
from systems.data import chunks

SAMPLE_RATE = 32000
VALIDATION_FRACTION = 0.1
SPLIT_SEED = 42
PROBABILITY_THRESHOLD = 0.5


def resample(waveform, sample_rate, target_rate=SAMPLE_RATE):
    """Return the waveform at the target rate, by linear interpolation between samples."""
    if sample_rate == target_rate:
        return waveform
    target_length = int(round(len(waveform) * target_rate / sample_rate))
    if target_length <= 1:
        return np.zeros(1, dtype=np.float32)
    source_positions = np.linspace(0.0, 1.0, len(waveform), dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, waveform).astype(np.float32)


def training_files(size):
    """Return the joined training files of one training size, in name order.

    Each file is a dictionary with its name, corpus, audio path and annotations."""
    directory = config.prepared_dir(f"train_{size}_joined")
    files = []
    for label_path in sorted(directory.glob("*.json")):
        labels = json.loads(label_path.read_text())
        files.append({
            "name": label_path.stem,
            "corpus": data.corpus_of(label_path.stem),
            "audio_path": label_path.with_suffix(".wav"),
            "onsets": labels["onset"],
            "offsets": labels["offset"],
        })
    return files


def corpus_generator(corpus):
    """Return the random generator that draws the validation files of one corpus."""
    digest = hashlib.blake2b(f"{SPLIT_SEED}:{corpus}".encode(), digest_size=4).digest()
    return np.random.RandomState(int.from_bytes(digest, "big") % (2 ** 31))


def split_training_files(files):
    """Split the training files into (training, validation), corpus by corpus.

    A tenth of the files of every corpus, at least one, is held out for validation. A corpus
    with a single file is used for training only."""
    names_of_corpus = defaultdict(set)
    for file in files:
        names_of_corpus[file["corpus"]].add(file["name"])
    validation_names = set()
    for corpus in sorted(names_of_corpus):
        names = sorted(names_of_corpus[corpus])
        if len(names) < 2:
            continue
        corpus_generator(corpus).shuffle(names)
        count = max(1, int(round(VALIDATION_FRACTION * len(names))))
        count = min(count, len(names) - 1)
        validation_names.update(names[:count])
    training = []
    validation = []
    for file in files:
        if file["name"] in validation_names:
            validation.append(file)
        else:
            training.append(file)
    return training, validation


def training_waveform(file):
    """Return the audio of a training file at 32 kHz."""
    waveform, sample_rate = soundfile.read(file["audio_path"], dtype="float32", always_2d=True)
    return resample(waveform.mean(axis=1), sample_rate)


def test_waveforms():
    """Yield (chunk_id, waveform at 32 kHz) for every test chunk, in dataset order."""
    for dataset_file in chunks.dataset_files("test"):
        for row in chunks.rows_with_audio(dataset_file):
            waveform, sample_rate = chunks.decode_audio(row)
            yield row["chunk_id"], resample(waveform, sample_rate)


def vocalizations(probability, frame_rate):
    """Return [(onset, offset)] in seconds: the runs of frames with probability >= 0.5.

    No gaps are filled and no short runs are removed."""
    vocal = probability >= PROBABILITY_THRESHOLD
    if not vocal.any():
        return []
    change = np.diff(vocal.astype(np.int8))
    starts = list(np.where(change == 1)[0] + 1)
    ends = list(np.where(change == -1)[0] + 1)
    if vocal[0]:
        starts = [0] + starts
    if vocal[-1]:
        ends = ends + [len(vocal)]
    detected = []
    for start, end in zip(starts, ends):
        detected.append((int(start) / frame_rate, int(end) / frame_rate))
    return detected


def write_predictions(system, size, rows):
    """Write predictions/<system>/detection_<size>.json.gz."""
    path = config.prediction_file(system, "detection", size)
    data.write_json(path, rows)
    print(f"wrote {path}: {len(rows)} chunks")
