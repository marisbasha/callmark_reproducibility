"""Join the training windows into files of at least 30 s: prepared/train_<size>_joined/.

    python -m systems.data.join_windows --data DIR

The windows of prepared/train_<size> that share corpus, sampling rate and spectrogram parameters
are concatenated in name order, without gaps. A file is closed once it reaches 30 s; a shorter
remainder is appended to the last file of its group. Audio and annotations are unchanged.
Files are named <corpus>__g<group>_cat<number>.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import soundfile

from benchmark import config, data
from systems.data import chunks

TARGET_LENGTH = 30.0


def group_windows(directory):
    """Return the label files of the windows grouped by corpus and spectrogram parameters."""
    groups = defaultdict(list)
    for label_file in sorted(directory.glob("*.json")):
        labels = data.read_json(label_file)
        key = (
            data.corpus_of(label_file.name),
            labels["sr"],
            labels["spec_time_step"],
            labels["min_frequency"],
        )
        groups[key].append(label_file)
    return groups


def split_into_files(label_files):
    """Return the windows of one group as lists, one list per joined file."""
    batches = []
    current = []
    current_length = 0.0
    for label_file in label_files:
        current.append(label_file)
        current_length += soundfile.info(label_file.with_suffix(".wav")).duration
        if current_length >= TARGET_LENGTH:
            batches.append(current)
            current = []
            current_length = 0.0
    if current:
        if batches:
            batches[-1].extend(current)
        else:
            batches.append(current)
    return batches


def join(label_files):
    """Return (waveform, sample rate, labels) of the concatenation of some windows."""
    waveforms = []
    onsets = []
    offsets = []
    taxa = []
    elapsed = 0.0
    for label_file in label_files:
        labels = data.read_json(label_file)
        waveform, sample_rate = soundfile.read(label_file.with_suffix(".wav"), dtype="int16")
        duration = len(waveform) / sample_rate
        for onset, offset in zip(labels["onset"], labels["offset"]):
            onsets.append(round(elapsed + onset, 4))
            offsets.append(round(elapsed + min(offset, duration), 4))
        taxa.extend(labels["cluster"])
        waveforms.append(waveform)
        elapsed += duration
    joined = data.read_json(label_files[0])
    joined["onset"] = onsets
    joined["offset"] = offsets
    joined["cluster"] = taxa
    return np.concatenate(waveforms), sample_rate, joined


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(config.SIZES))
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    for size in arguments.sizes:
        groups = group_windows(config.prepared_dir(f"train_{size}"))
        directory = config.prepared_dir(f"train_{size}_joined")
        directory.mkdir(parents=True, exist_ok=True)
        written = 0
        for group_number, key in enumerate(sorted(groups)):
            batches = split_into_files(groups[key])
            for file_number, label_files in enumerate(batches):
                waveform, sample_rate, labels = join(label_files)
                name = f"{key[0]}__g{group_number:02d}_cat{file_number:03d}"
                chunks.write_pair(directory, name, waveform, sample_rate, labels)
                written += 1
        print(f"size {size}: {written} files in {len(groups)} groups -> {directory}")


if __name__ == "__main__":
    main()
