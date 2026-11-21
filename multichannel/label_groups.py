"""Label unannotated BirdPark group recordings with MC-WhisperSeg.

MC-WhisperSeg is run once per focal bird on a whole recording; the detected vocalizations of
all birds are pooled and stored with the microphone signal as one training chunk of the
microphone-only detector:

    group_audio/<experiment>__<recording>_as_target_channel_1.wav    microphone, 16 kHz
    group_audio/<experiment>__<recording>_as_target_channel_1.json   onset, offset, bird; the bird
                                                                     names and the root mean square
                                                                     of every accelerometer channel

    python label_groups.py RECORDINGS [--shard I --of N]

RECORDINGS holds <experiment>/.../<recording>.h5 in the format of the public BirdPark
recordings. Recordings of the test set are refused. --shard I --of N labels every N-th
recording, for one process per GPU.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np
import soundfile

from data import SAMPLE_RATE, data_root, load_public, public_info, public_recordings
from infer import DECODING, model_inputs

sys.path.insert(0, str(Path(__file__).resolve().parent / "mcwhisperseg"))
from model import WhisperSegmenterFast  # noqa: E402

CHUNK_FIELDS = {
    "species": "animal",
    "sr": SAMPLE_RATE,
    "min_frequency": 0,
    "spec_time_step": 0.0025,
    "min_segment_length": 0.001,
    "tolerance": 0.01,
    "time_per_frame_for_scoring": 0.001,
    "eps": 0.02,
}


def recordings_of_shard(folder, shard, n_shards):
    """Paths of the recordings in the folder that belong to the shard, ordered by path.
    A recording is assigned to a shard by the checksum of its file name."""
    paths = []
    for path in sorted(folder.rglob("*.h5")):
        if zlib.crc32(path.name.encode()) % n_shards == shard:
            paths.append(path)
    return paths


def detect_vocalizations(segmenter, n_birds, accelerometers, microphone):
    """Onset, offset and bird index of the vocalizations detected for every focal bird,
    ordered by onset."""
    onsets = []
    offsets = []
    birds = []
    inputs = model_inputs("multichannel", n_birds, accelerometers, microphone)
    for bird, audio in enumerate(inputs):
        found = segmenter.segment(audio, SAMPLE_RATE, **DECODING)
        for onset in found["onset"]:
            onsets.append(float(onset))
            birds.append(bird)
        for offset in found["offset"]:
            offsets.append(float(offset))
    order = np.argsort(onsets)
    sorted_onsets = [onsets[i] for i in order]
    sorted_offsets = [offsets[i] for i in order]
    sorted_birds = [birds[i] for i in order]
    return sorted_onsets, sorted_offsets, sorted_birds


def root_mean_square(signal):
    """Root mean square of a signal, computed in double precision."""
    return float(np.sqrt(np.mean(np.asarray(signal, dtype=np.float64) ** 2)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", type=Path)
    parser.add_argument("--data", help="data root (default: $CALLMARK_DATA)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    args = parser.parse_args()
    root = data_root(args.data)
    out = root / "group_audio"
    out.mkdir(exist_ok=True)

    test_names = set()
    for path, _ in public_recordings(root):
        test_names.add(path.name)
    paths = recordings_of_shard(args.recordings, args.shard, args.of)
    for path in paths:
        if path.name in test_names:
            raise SystemExit("RECORDINGS contains test recordings")
    segmenter = WhisperSegmenterFast(str(root / "models/multichannel/ct2"), device="cuda")

    for path in paths:
        experiment = path.relative_to(args.recordings).parts[0]
        stem = f"{experiment}__{path.stem}_as_target_channel_1"
        if (out / f"{stem}.json").exists():
            continue
        bird_names, _ = public_info(path)
        with h5py.File(path) as file:
            n_samples = file["/radioSignals"].shape[1]
        accelerometers, microphone = load_public(path, n_samples)
        onsets, offsets, birds = detect_vocalizations(
            segmenter, len(bird_names), accelerometers, microphone)
        accelerometer_rms = []
        for channel in accelerometers:
            accelerometer_rms.append(root_mean_square(channel))
        soundfile.write(out / f"{stem}.wav", microphone, SAMPLE_RATE)
        label = {
            "onset": onsets,
            "offset": offsets,
            "cluster": ["Vocal"] * len(onsets),
            "bird": birds,
            **CHUNK_FIELDS,
            "birds": bird_names,
            "accelerometer_rms": accelerometer_rms,
        }
        (out / f"{stem}.json").write_text(json.dumps(label))
        summary = f"{len(bird_names)} birds, {len(onsets)} vocalizations"
        print(f"{experiment} {path.stem}: {summary}", flush=True)


if __name__ == "__main__":
    main()
