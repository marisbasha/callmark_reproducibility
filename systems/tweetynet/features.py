"""Spectrogram input of TweetyNet, the model and its frame-wise vocal probability."""
import numpy as np
import torch

from benchmark import config
from systems.das import data

FREQUENCY_BANDS = 128
HOP_SECONDS = 0.005
WINDOW_FRAMES = 176

# Windows shorter than this at the end of a chunk are not passed through the model.
SHORTEST_WINDOW_FRAMES = 8


def model_file(size):
    """Return models/tweetynet/<size>/best.pt of the data root."""
    return config.DATA / "models" / "tweetynet" / str(size) / "best.pt"


def build_model():
    """Return an untrained TweetyNet with two output classes (noise, vocal)."""
    # triton is imported before vak: with PyTorch 2.13 the import of vak otherwise crashes.
    try:
        import triton  # noqa: F401
    except ImportError:
        pass
    from vak.nets import TweetyNet
    return TweetyNet(num_classes=2, num_freqbins=FREQUENCY_BANDS)


def band_filters(frequencies, sample_rate):
    """Return 128 triangular filters with centres spaced evenly from 0 Hz to half the rate."""
    edges = np.linspace(0, sample_rate // 2, FREQUENCY_BANDS + 2)
    filters = np.zeros((FREQUENCY_BANDS, len(frequencies)), dtype=np.float32)
    for band in range(FREQUENCY_BANDS):
        low, centre, high = edges[band], edges[band + 1], edges[band + 2]
        rising = (frequencies - low) / max(centre - low, 1e-9)
        falling = (high - frequencies) / max(high - centre, 1e-9)
        filters[band] = np.maximum(0, np.minimum(rising, falling))
    return filters


def spectrogram(waveform, sample_rate=data.SAMPLE_RATE):
    """Return (log band energies of shape (128, frames), frame length in seconds).

    The median over time is subtracted from every band."""
    hop = max(1, int(round(HOP_SECONDS * sample_rate)))
    window_length = 1 << int(np.ceil(np.log2(max(hop * 4, 64))))
    waveform = np.ascontiguousarray(waveform)
    if len(waveform) < window_length:
        waveform = np.pad(waveform, (0, window_length - len(waveform)))
    frame_count = 1 + (len(waveform) - window_length) // hop
    window = np.hanning(window_length).astype(np.float32)
    sample_bytes = waveform.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        waveform, shape=(window_length, frame_count), strides=(sample_bytes, hop * sample_bytes)
    ).copy()
    power = np.abs(np.fft.rfft(frames * window[:, None], axis=0)) ** 2
    frequencies = np.fft.rfftfreq(window_length, 1 / sample_rate)
    band_energy = band_filters(frequencies, sample_rate) @ power
    log_energy = np.log(band_energy + 1e-10).astype(np.float32)
    log_energy = log_energy - np.median(log_energy, axis=1, keepdims=True)
    return log_energy, hop / sample_rate


def vocal_probability(model, log_energy, device):
    """Return the vocal probability of every frame of one chunk.

    The spectrogram is passed through the model in consecutive windows of 176 frames."""
    frame_count = log_energy.shape[1]
    probability = np.zeros(frame_count, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, frame_count, WINDOW_FRAMES):
            window = log_energy[:, start:start + WINDOW_FRAMES]
            if window.shape[1] < SHORTEST_WINDOW_FRAMES:
                continue
            batch = torch.from_numpy(window[None, None]).to(device)
            window_probability = torch.softmax(model(batch), 1)[0, 1].cpu().numpy()
            probability[start:start + len(window_probability)] = window_probability
    return probability
