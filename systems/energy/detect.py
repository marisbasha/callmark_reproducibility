"""Energy-threshold detector: vocalizations in the test chunks without any training.

    python -m systems.energy.detect --data DIR

The frame energy of a chunk is its mean spectrogram level in the frequency band of the corpus,
after the median over time has been subtracted from every frequency. A vocalization starts
where the energy reaches the median plus 3 robust standard deviations of the chunk and ends
where it falls below the median plus 1. Vocalizations less than 30 ms apart are merged, and
those shorter than 20 ms are removed. Writes predictions/energy/detection.json.gz.
"""
import argparse

import numpy as np

from benchmark import config, data
from systems.data import chunks

# Frequency band of each corpus in Hz; the upper limit is at most half the sampling rate.
FREQUENCY_BAND = {
    "budgie": (0, 10000),
    "canary": (0, 22050),
    "enabirds": (0, 16000),
    "infant_marmosets": (0, 22050),
    "nbm": (0, 22050),
    "ozf": (0, 22050),
    "right_whale": (0, 600),
    "rodent_usv": (15000, 125000),
    "singing_mice": (10000, 125000),
    "sne_full": (0, 16000),
    "spiders": (0, 8000),
    "tworadiobirds": (0, 12000),
    "wytham": (0, 11025),
}

WINDOW_SAMPLES = 1024
HOP_SAMPLES = WINDOW_SAMPLES // 4
START_DEVIATIONS = 3.0
END_DEVIATIONS = 1.0
MERGE_GAP_SECONDS = 0.03
SHORTEST_SECONDS = 0.02


def frame_energy(waveform, sample_rate, corpus):
    """Return the energy of every spectrogram frame in dB relative to the median spectrum.

    Returns None for audio shorter than two frames."""
    frame_count = 1 + max(0, (len(waveform) - WINDOW_SAMPLES) // HOP_SAMPLES)
    if frame_count < 2:
        return None
    sample_bytes = waveform.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(WINDOW_SAMPLES, frame_count),
        strides=(sample_bytes, HOP_SAMPLES * sample_bytes),
    ).copy()
    window = np.hanning(WINDOW_SAMPLES)
    power = np.abs(np.fft.rfft(frames * window[:, None], axis=0)) ** 2
    frequencies = np.fft.rfftfreq(WINDOW_SAMPLES, 1 / sample_rate)
    lowest, highest = FREQUENCY_BAND[corpus]
    highest = min(highest, sample_rate / 2)
    in_band = (frequencies >= lowest) & (frequencies <= highest)
    level = 10 * np.log10(power[in_band] + 1e-12)
    level = level - np.median(level, axis=1, keepdims=True)
    return level.mean(axis=0)


def runs_above_threshold(energy, start_threshold, end_threshold):
    """Return [(first frame, end frame)]: a run starts at the first frame that reaches the
    start threshold and ends at the first frame below the end threshold."""
    runs = []
    start = None
    for frame, value in enumerate(energy):
        if start is None and value >= start_threshold:
            start = frame
        elif start is not None and value < end_threshold:
            runs.append((start, frame))
            start = None
    if start is not None:
        runs.append((start, len(energy)))
    return runs


def detect(waveform, sample_rate, corpus):
    """Return [(onset, offset)] in seconds from the start of the waveform."""
    energy = frame_energy(waveform, sample_rate, corpus)
    if energy is None:
        return []
    median = np.median(energy)
    # 1.4826 x the median absolute deviation estimates the standard deviation of a normal
    # distribution.
    deviation = np.median(np.abs(energy - median)) * 1.4826
    if deviation <= 0:
        return []
    start_threshold = median + START_DEVIATIONS * deviation
    end_threshold = median + END_DEVIATIONS * deviation
    frame_seconds = HOP_SAMPLES / sample_rate

    merged = []
    for start, end in runs_above_threshold(energy, start_threshold, end_threshold):
        onset = start * frame_seconds
        offset = end * frame_seconds
        if merged and onset - merged[-1][1] < MERGE_GAP_SECONDS:
            merged[-1] = (merged[-1][0], offset)
        else:
            merged.append((onset, offset))
    detected = []
    for onset, offset in merged:
        if offset - onset >= SHORTEST_SECONDS:
            detected.append((onset, offset))
    return detected


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    rows = []
    for dataset_file in chunks.dataset_files("test"):
        for row in chunks.rows_with_audio(dataset_file):
            waveform, sample_rate = chunks.decode_audio(row)
            detected = detect(waveform, sample_rate, row["dataset"])
            rows.append({"chunk_id": row["chunk_id"], "pred": detected})
    path = config.prediction_file("energy", "detection")
    data.write_json(path, rows)
    print(f"wrote {path}: {len(rows)} chunks")


if __name__ == "__main__":
    main()
