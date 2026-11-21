"""Write the detection test set: prepared/test/<chunk_id>.wav and .json for every test chunk.

    python -m systems.data.prepare_test --data DIR

The audio is the chunk at its own sampling rate as 16-bit WAV. The JSON file holds the
spectrogram parameters of the corpus and the annotations, with offsets clipped to the end of the
audio and times rounded to a microsecond.
"""
import argparse
from multiprocessing import Pool

from benchmark import config
from systems.data import chunks


def test_labels(row, corpus, sample_rate, duration):
    """Return the content of the JSON file of one test chunk."""
    onsets = []
    offsets = []
    for onset, offset in zip(row["onsets"], row["offsets"]):
        onset = max(0.0, float(onset))
        offset = min(duration, float(offset))
        if offset > onset:
            onsets.append(round(onset, 6))
            offsets.append(round(offset, 6))
    labels = chunks.spectrogram_parameters(corpus, sample_rate)
    labels["onset"] = onsets
    labels["offset"] = offsets
    labels["cluster"] = ["Vocal"] * len(onsets)
    labels["species"] = corpus
    return labels


def write_file(arguments):
    """Write the chunks of one dataset file and return how many were written."""
    dataset_file, directory = arguments
    written = 0
    for row in chunks.rows_with_audio(dataset_file):
        waveform, sample_rate = chunks.decode_audio(row)
        duration = len(waveform) / sample_rate
        labels = test_labels(row, row["dataset"], sample_rate, duration)
        chunks.write_pair(directory, row["chunk_id"], waveform, sample_rate, labels)
        written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    parser.add_argument("--workers", type=int, default=12, help="parallel processes")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    directory = config.prepared_dir("test")
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset_file in chunks.dataset_files("test"):
        jobs.append((dataset_file, directory))
    with Pool(arguments.workers) as pool:
        written = pool.map(write_file, jobs)
    print(f"{sum(written)} chunks -> {directory}")


if __name__ == "__main__":
    main()
