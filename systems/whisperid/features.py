"""Model input of WhisperID: a 5-s log-mel spectrogram with the vocalization marked."""
import functools
import sys

import numpy as np

from systems.whisperseg import common as whisperseg

sys.path.insert(0, str(whisperseg.WHISPERSEG))
from audio_utils import WhisperSegFeatureExtractor  # noqa: E402

WINDOW_SECONDS = 5.0
COLUMN_SECONDS = 0.01
COLUMNS = 500
# The encoder halves the time resolution.
ENCODER_FRAMES = 250
# The marker sets this many of the highest mel bands to the maximum of the spectrogram.
MARKER_BANDS = 4
# The lowest mel band starts at this frequency.
MIN_FREQUENCY = 80


@functools.lru_cache(maxsize=None)
def extractor(sampling_rate):
    """Return the spectrogram extractor of one sampling rate."""
    return WhisperSegFeatureExtractor(sampling_rate, COLUMN_SECONDS, MIN_FREQUENCY)


def spectrogram(waveform, sampling_rate):
    """Return the 80 x 500 log-mel spectrogram of a waveform.

    Longer audio is cut and shorter audio is padded with the lowest value of the spectrogram."""
    extracted = extractor(sampling_rate)(
        waveform, sampling_rate=sampling_rate, padding="do_not_pad"
    )
    bands = extracted["input_features"][0][:, :COLUMNS]
    missing = COLUMNS - bands.shape[1]
    if missing > 0:
        padding = np.full((bands.shape[0], missing), float(bands.min()), dtype=bands.dtype)
        bands = np.concatenate([bands, padding], axis=1)
    return bands.astype(np.float32)


def mark(bands, onset, offset):
    """Mark the columns between onset and offset (seconds) in the highest mel bands."""
    first = max(0, int(onset / COLUMN_SECONDS))
    end = min(COLUMNS, int(np.ceil(offset / COLUMN_SECONDS)))
    if end > first:
        bands[-MARKER_BANDS:, first:end] = bands.max()
    return bands


def augmented(waveform, generator):
    """Return a training waveform with random gain, added noise and random polarity."""
    waveform = waveform * float(10 ** (generator.uniform(-6, 6) / 20))
    signal_to_noise = generator.uniform(20, 40)
    if len(waveform):
        level = float(np.sqrt(np.mean(waveform ** 2)))
    else:
        level = 0.0
    if level > 0:
        noise_level = level / (10 ** (signal_to_noise / 20))
        noise = generator.normal(0, noise_level, len(waveform))
        waveform = waveform + noise.astype(np.float32)
    if generator.random() < 0.5:
        waveform = -waveform
    return np.clip(waveform, -1.0, 1.0).astype(np.float32)


def example_features(example, waveform, shift_seconds=0.0):
    """Return the marked spectrogram of an example.

    A shift rotates the audio circularly by that many seconds; the marker moves with it."""
    sampling_rate = example["sampling_rate"]
    shift = 0.0
    if shift_seconds and len(waveform) > 8:
        samples = int(round(shift_seconds * sampling_rate))
        waveform = np.roll(waveform, samples)
        shift = samples / float(sampling_rate)
    bands = spectrogram(waveform, sampling_rate)
    onset = example["onset_in_window"] + shift
    offset = example["offset_in_window"] + shift
    return mark(bands, onset, offset)


def training_features(example, waveform, generator):
    """Return the marked spectrogram of an augmented training example.

    After the waveform augmentation, the audio is rotated circularly by up to a tenth of its
    length; the marker moves with it."""
    sampling_rate = example["sampling_rate"]
    waveform = augmented(waveform, generator)
    shift = 0.0
    if len(waveform) > 8:
        samples = int(generator.uniform(-0.1, 0.1) * len(waveform))
        waveform = np.roll(waveform, samples)
        shift = samples / float(sampling_rate)
    bands = spectrogram(waveform, sampling_rate)
    onset = example["onset_in_window"] + shift
    offset = example["offset_in_window"] + shift
    return mark(bands, onset, offset)


def prompt(sampling_rate):
    """Return the decoder prompt; the taxon name follows it after a space."""
    return f"sr: {sampling_rate // 1000} kHz, species:"
