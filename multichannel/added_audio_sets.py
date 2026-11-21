"""Training sets of the microphone-only detector with added group audio.

For each set of added_audio_sets.json (7h, 30h, 91h) this writes a data root

    $CALLMARK_DATA/added_audio/<set>/multichannel/
        train_microphone/   the 88 chunks of pair recordings and the group recordings of the set
        val_microphone/     the validation set of the pair recordings
        test_public/        the public test recordings

as links, so that train.py and infer.py run on it with --data $CALLMARK_DATA/added_audio/<set>.
The group recordings are the chunks written by label_groups.py.

    python added_audio_sets.py

A recording belongs to a set only if every bird of the group carried a working transmitter:
a bird without one is heard on the microphone without being labelled. In the sets 30h and 91h
a transmitter counts as failed when its accelerometer channel is noisy and MC-WhisperSeg found
almost no vocalizations of that bird (failed_transmitters); the script checks this.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data import data_root

SETS = json.loads((Path(__file__).resolve().parent / "added_audio_sets.json").read_text())
NOISY_RMS = 0.01
MIN_VOCALIZATIONS = 3


def failed_transmitters(label):
    """Birds whose accelerometer channel is noisy (root mean square of the normalized,
    high-pass filtered signal at least NOISY_RMS) with fewer than MIN_VOCALIZATIONS detected."""
    bird_names = label["birds"]
    bird_of_vocalization = np.array(label["bird"], dtype=int)
    n_vocalizations = np.bincount(bird_of_vocalization, minlength=len(bird_names))
    failed = []
    for name, rms, count in zip(bird_names, label["accelerometer_rms"], n_vocalizations):
        if rms >= NOISY_RMS and count < MIN_VOCALIZATIONS:
            failed.append(name)
    return failed


def link_pair_data(source, root):
    """Link the validation set, the test recordings and every training chunk of the pair
    recordings from the data root source into the new data root."""
    (root / "train_microphone").mkdir(parents=True)
    for folder in ("val_microphone", "test_public"):
        (root / folder).symlink_to(source / folder)
    for file in sorted((source / "train_microphone").iterdir()):
        (root / "train_microphone" / file.name).symlink_to(file.resolve())


def link_group_recording(source, root, recording):
    """Link the audio and label of one group recording into the training folder, after
    checking that no transmitter of the group failed."""
    chunk = source / "group_audio" / f"{recording}_as_target_channel_1"
    label = json.loads(chunk.with_suffix(".json").read_text())
    if "accelerometer_rms" in label and failed_transmitters(label):
        raise SystemExit(f"{recording}: failed transmitter")
    for suffix in (".wav", ".json"):
        link = root / "train_microphone" / (chunk.name + suffix)
        link.symlink_to(chunk.with_suffix(suffix).resolve())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="data root (default: $CALLMARK_DATA)")
    args = parser.parse_args()
    source = data_root(args.data).resolve()

    for name, recordings in SETS.items():
        root = source.parent / "added_audio" / name / "multichannel"
        link_pair_data(source, root)
        for recording in recordings:
            link_group_recording(source, root, recording)
        print(f"{name}: {len(recordings)} group recordings added, {root}")


if __name__ == "__main__":
    main()
