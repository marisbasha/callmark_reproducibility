"""Detect vocalizations with one of the three systems.

  microphone     one pass over the shared microphone; returns the vocalizations of all birds
  accelerometer  one pass per bird over that bird's accelerometer
  multichannel   one pass per bird over three channels: the accelerometer of the focal bird,
                 the summed accelerometers of the other birds, and the microphone

Sets: test_lab (16 held-out recordings of pairs), test_public (16 public recordings of two,
four and eight birds) and val (validation examples of the training collection).
Writes one JSON per recording to predictions/<system>/<set>/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from data import SAMPLE_RATE, SYSTEMS, data_root, lab_recordings, load_lab, load_public, \
    public_info, public_recordings, read_wav

sys.path.insert(0, str(Path(__file__).resolve().parent / "mcwhisperseg"))
from model import WhisperSegmenterFast  # noqa: E402

DECODING = dict(min_frequency=0, spec_time_step=0.0025, min_segment_length=0.005, eps=0.02,
                num_trials=1, num_beams=4, max_length=448, batch_size=4)


def recordings(root, system, test_set):
    """Yield (recording name, channel names, accelerometers, microphone)."""
    if test_set == "test_lab":
        for path in lab_recordings(root):
            accelerometers, microphone = load_lab(path)
            yield path.stem, ["accelerometer1", "accelerometer2"], accelerometers, microphone
    elif test_set == "test_public":
        for path, n_samples in public_recordings(root):
            bird_names, _ = public_info(path)
            accelerometers, microphone = load_public(path, n_samples)
            yield path.stem, bird_names, accelerometers, microphone
    elif system == "microphone":
        for path in sorted((root / "val_microphone").glob("*.wav")):
            yield path.stem, [], None, read_wav(path)
    else:
        # A validation example is a (focal, other, microphone) triple labelled for the
        # focal bird only, stored as <example>_as_target_channel_{1,2,3}.wav.
        for path in sorted((root / "val").glob("*_as_target_channel_1.wav")):
            prefix = str(path)[:-len("1.wav")]
            focal = read_wav(f"{prefix}1.wav")
            other = read_wav(f"{prefix}2.wav")
            microphone = read_wav(f"{prefix}3.wav")
            yield path.stem, ["focal"], np.stack([focal, other]), microphone


def model_inputs(system, n_birds, accelerometers, microphone):
    """One (channels, samples) array per decoding pass."""
    if system == "microphone":
        return [microphone[None]]
    inputs = []
    if system == "accelerometer":
        for bird in range(n_birds):
            inputs.append(accelerometers[bird][None])
        return inputs
    for bird in range(n_birds):
        others = np.zeros_like(accelerometers[bird])
        for other in range(len(accelerometers)):
            if other != bird:
                others += accelerometers[other]
        inputs.append(np.stack([accelerometers[bird], others, microphone]))
    return inputs


def as_floats(values):
    """The values as a list of Python floats."""
    return [float(value) for value in values]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--set", choices=["test_lab", "test_public", "val"], required=True)
    parser.add_argument("--data", help="data root (default: $CALLMARK_DATA)")
    parser.add_argument(
        "--model", type=Path, help="CTranslate2 model (default: models/<system>/ct2)")
    parser.add_argument(
        "--out", type=Path, help="output folder (default: predictions/<system>/<set>)")
    args = parser.parse_args()
    root = data_root(args.data)
    out = args.out or root / "predictions" / args.system / args.set
    out.mkdir(parents=True, exist_ok=True)

    model_folder = args.model or root / "models" / args.system / "ct2"
    segmenter = WhisperSegmenterFast(str(model_folder), device="cuda")

    for name, channels, accelerometers, microphone in recordings(root, args.system, args.set):
        predictions = []
        counts = []
        for audio in model_inputs(args.system, len(channels), accelerometers, microphone):
            found = segmenter.segment(audio, SAMPLE_RATE, **DECODING)
            onsets = as_floats(found["onset"])
            offsets = as_floats(found["offset"])
            predictions.append({"onset": onsets, "offset": offsets})
            counts.append(len(onsets))
        if args.system == "microphone":
            channels = ["microphone"]
        output = {"channels": channels, "predictions": predictions}
        (out / f"{name}.json").write_text(json.dumps(output))
        print(args.set, name, counts, flush=True)


if __name__ == "__main__":
    main()
