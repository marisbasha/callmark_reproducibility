"""Data root, recordings and reference vocalizations of the multi-channel experiments."""
import json
import os
from pathlib import Path

import h5py
import librosa
import numpy as np
import soundfile
from scipy.signal import butter, sosfiltfilt

SAMPLE_RATE = 16000
SYSTEMS = ("microphone", "accelerometer", "multichannel")
COHORTS = ("lab_pairs", "public_pairs", "four_birds", "eight_birds", "public_groups")
PUBLIC_COHORT = {2: "public_pairs", 4: "four_birds", 8: "eight_birds"}
MICROPHONE_PEAK = 0.2


def data_root(argument=None):
    """Folder of the multi-channel data, from --data or the variable CALLMARK_DATA."""
    root = argument or os.environ.get("CALLMARK_DATA")
    if not root:
        raise SystemExit("Set CALLMARK_DATA or pass --data")
    return Path(root) / "multichannel"


def normalize(accelerometers, microphone):
    """Remove the mean of every channel, divide all accelerometers by their common
    peak, which keeps their relative amplitudes, and scale the microphone to a fixed peak."""
    accelerometers = accelerometers - accelerometers.mean(axis=1, keepdims=True)
    microphone = microphone - microphone.mean()
    accelerometer_peak = max(float(np.abs(accelerometers).max()), 1e-9)
    microphone_peak = max(float(np.abs(microphone).max()), 1e-9)
    accelerometers = accelerometers / accelerometer_peak
    microphone = microphone / microphone_peak * MICROPHONE_PEAK
    return accelerometers, microphone


def read_wav(path):
    """Samples of a 16 kHz audio file as 32-bit floats."""
    audio, rate = soundfile.read(path, dtype="float32")
    assert rate == SAMPLE_RATE
    return audio


def lab_recordings(root):
    """Paths of the held-out pair recordings, ordered by name."""
    return sorted((root / "test_lab").glob("*.wav"))


def load_lab(path):
    """A held-out pair recording: channels are accelerometer 1, accelerometer 2, microphone."""
    audio = read_wav(path).T
    return normalize(audio[:2], audio[2])


def lab_reference(path):
    """{bird index: [(onset, offset), ...]} in seconds."""
    label = json.loads(path.with_suffix(".json").read_text())
    reference = {}
    for onset, offset, bird in zip(label["onset"], label["offset"], label["channelIndex"]):
        reference.setdefault(int(bird), []).append((float(onset), float(offset)))
    return reference


def recording_name(recording):
    """File name without extension of a (path, number of annotated samples) pair."""
    path, _ = recording
    return path.stem


def public_recordings(root):
    """Annotated public recordings as (path, number of annotated samples), ordered by name."""
    with h5py.File(root / "test_public/segments.h5") as file:
        episodes = file["episodes"][:]
    recordings = []
    for episode in episodes:
        path = root / "test_public" / episode["filename"].decode()
        recordings.append((path, int(episode["offset"])))
    return sorted(recordings, key=recording_name)


def public_info(path):
    """Bird name of every accelerometer channel and the stored sampling rate."""
    with h5py.File(path) as file:
        info = file["/recInfo"].attrs
        bird_names = [str(name) for name in info["radioChNames"]]
        rate = float(info["sampleRate_Hz"])
    return bird_names, rate


def load_public(path, n_samples, highpass=250.0):
    """Annotated part of a public recording at 16 kHz. The stored accelerometer signals
    carry body-movement energy at low frequencies and are high-pass filtered
    (second-order Butterworth, forward and backward) before normalization."""
    _, rate = public_info(path)
    with h5py.File(path) as file:
        accelerometers = np.asarray(file["/radioSignals"][:, :n_samples], dtype=np.float32)
        microphone = np.asarray(file["/daqSignals"][0, :n_samples], dtype=np.float32)
    sos = butter(2, highpass, btype="highpass", fs=rate, output="sos")
    accelerometers = sosfiltfilt(sos, accelerometers, axis=1).astype(np.float32)
    accelerometers, microphone = normalize(accelerometers, microphone)
    resampled = []
    for channel in accelerometers:
        resampled.append(librosa.resample(channel, orig_sr=rate, target_sr=SAMPLE_RATE))
    microphone = librosa.resample(microphone, orig_sr=rate, target_sr=SAMPLE_RATE)
    return np.stack(resampled), microphone


def public_segments(root, path, kind):
    """Annotations of one recording as (channel or bird name, onset, offset) in seconds.
    kind: 'consolidated', 'transmitter-based' or 'microphone-based'. Sample indices are
    one-based."""
    _, rate = public_info(path)
    filename = f"{path.parent.name}/{path.name}"
    name_field = "bird" if kind == "consolidated" else "channel"
    with h5py.File(root / "test_public/segments.h5") as file:
        segments = file["segments"][:]
    selected = []
    for segment in segments:
        if segment["segType"].decode() != kind or segment["filename"].decode() != filename:
            continue
        onset = (float(segment["onset"]) - 1) / rate
        offset = (float(segment["offset"]) - 1) / rate
        selected.append((segment[name_field].decode(), onset, offset))
    return selected


def public_reference(root, path):
    """Consolidated reference as {caller: [(onset, offset), ...]}. Callers are accelerometer
    channel indices; a vocalization not assigned to a bird gets a caller of its own."""
    bird_names, _ = public_info(path)
    reference = {}
    unassigned = {}
    for bird, onset, offset in public_segments(root, path, "consolidated"):
        if bird in bird_names:
            reference.setdefault(bird_names.index(bird), []).append((onset, offset))
        else:
            unassigned[f"unassigned{len(unassigned)}"] = [(onset, offset)]
    reference.update(unassigned)
    return reference


def test_recordings(root):
    """(cohort, test set, recording name, reference) of the 32 test recordings."""
    recordings = []
    for path in lab_recordings(root):
        recordings.append(("lab_pairs", "test_lab", path.stem, lab_reference(path)))
    for path, _ in public_recordings(root):
        bird_names, _ = public_info(path)
        cohort = PUBLIC_COHORT[len(bird_names)]
        recordings.append((cohort, "test_public", path.stem, public_reference(root, path)))
    return recordings


def in_cohort(cohort, name):
    """Whether a recording of this cohort counts towards the cohort called name.
    public_groups joins the recordings of four and of eight birds."""
    if cohort == name:
        return True
    return name == "public_groups" and cohort in ("four_birds", "eight_birds")
