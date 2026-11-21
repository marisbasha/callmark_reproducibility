"""Build the published dataset from the original corpora.

    python -m systems.sources.build_dataset --data DIR

Reads corpora/ and writes dataset/classification/<split>-*.parquet (identification examples)
and dataset/segmentation/<split>-*.parquet (detection chunks) for the splits train_50,
train_100, train_500 and test. Audio is stored as 16-bit FLAC at the sampling rate of the
recording, the mean of the channels of a multi-channel recording.

Identification example: 5 s of audio centred on one vocalization, zero-padded beyond the ends
of the recording. The training split of size N holds the examples of rank below N of every
taxon. Detection chunk: the audio of one chunk with all its vocalizations. The test split holds
every chunk of the test recordings; the training split of size N holds the chunks with an
identification example of rank below N.
"""
import argparse
import io
import shutil

import numpy
import soundfile
from datasets import Audio, Dataset, Features, Sequence, Value

from benchmark import config
from systems.sources import build, corpora

WINDOW_SECONDS = 5.0
ROWS_PER_FILE = 1500
AUDIO_BYTES_PER_FILE = 400 * 1024 * 1024

# (name of the split, side of the recordings, training size; 0: all examples)
SPLITS = (("train_50", "train", 50), ("train_100", "train", 100), ("train_500", "train", 500),
          ("test", "test", 0))

IDENTIFICATION_COLUMNS = Features({
    "audio": Audio(),
    "sampling_rate": Value("int32"),
    "species": Value("string"),
    "taxon": Value("string"),
    "dataset": Value("string"),
    "dataset_name": Value("string"),
    "scorable": Value("bool"),
    "chunk_id": Value("string"),
    "rec_id": Value("string"),
    "event_index": Value("int32"),
    "onset_in_recording": Value("float32"),
    "offset_in_recording": Value("float32"),
    "onset_in_window": Value("float32"),
    "offset_in_window": Value("float32"),
    "duration": Value("float32"),
    "window_sec": Value("float32"),
    "fewshot_rank": Value("int32"),
    "split": Value("string"),
})

DETECTION_COLUMNS = Features({
    "audio": Audio(),
    "sampling_rate": Value("int32"),
    "dataset": Value("string"),
    "dataset_name": Value("string"),
    "chunk_id": Value("string"),
    "rec_id": Value("string"),
    "chunk_start": Value("float32"),
    "duration": Value("float32"),
    "n_events": Value("int32"),
    "onsets": Sequence(Value("float32")),
    "offsets": Sequence(Value("float32")),
    "fewshot_rank": Value("int32"),
    "split": Value("string"),
})


def read_window(path, centre):
    """Return (samples, sampling rate) of the window around `centre`, zero-padded outside the
    recording."""
    info = soundfile.info(path)
    rate = info.samplerate
    length = int(round(WINDOW_SECONDS * rate))
    first = int(round((centre - WINDOW_SECONDS / 2.0) * rate))
    read_from = max(0, first)
    read_to = min(info.frames, first + length)
    window = numpy.zeros(length, dtype=numpy.float32)
    if read_to <= read_from:
        return window, rate
    samples, _ = soundfile.read(path, start=read_from, stop=read_to, dtype="float32",
                                always_2d=True)
    samples = samples.mean(axis=1)
    window[read_from - first:read_from - first + len(samples)] = samples
    return window, rate


def read_chunk(chunk):
    """Return (samples, sampling rate) of the audio of a chunk."""
    info = soundfile.info(chunk["audio"])
    first = int(round(chunk["chunk_start"] * info.samplerate))
    last = min(info.frames, int(round(chunk["chunk_end"] * info.samplerate)))
    samples, rate = soundfile.read(chunk["audio"], start=first, stop=last, dtype="float32",
                                   always_2d=True)
    return samples.mean(axis=1), rate


def flac_bytes(samples, rate):
    """Return the samples encoded as 16-bit FLAC."""
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, rate, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def identification_rows(result, side, size):
    """Yield the identification examples of one split: taxa in sorted order, examples by rank."""
    chunk_of = {}
    for chunk in result["chunks"]:
        chunk_of[chunk["chunk_id"]] = chunk
    examples = result["examples"][side]
    for taxon in sorted(set(examples) - {build.UNIDENTIFIED}):
        for rank, (chunk_id, event_index) in enumerate(examples[taxon]):
            if size and rank >= size:
                continue
            chunk = chunk_of[chunk_id]
            event = chunk["events"][event_index]
            onset = chunk["chunk_start"] + event["onset"]
            offset = chunk["chunk_start"] + event["offset"]
            centre = (onset + offset) / 2.0
            window_start = centre - WINDOW_SECONDS / 2.0
            samples, rate = read_window(chunk["audio"], centre)
            yield {
                "audio": {"path": f"{chunk_id}__e{event_index:04d}.flac",
                          "bytes": flac_bytes(samples, rate)},
                "sampling_rate": rate,
                "species": taxon,
                "taxon": result["taxa"][taxon]["group"],
                "dataset": chunk["corpus"],
                "dataset_name": corpora.CORPUS_LABEL[chunk["corpus"]],
                "scorable": result["taxa"][taxon]["scorable"],
                "chunk_id": chunk_id,
                "rec_id": chunk["rec_id"],
                "event_index": event_index,
                "onset_in_recording": round(onset, 4),
                "offset_in_recording": round(offset, 4),
                "onset_in_window": round(onset - window_start, 4),
                "offset_in_window": round(offset - window_start, 4),
                "duration": round(offset - onset, 4),
                "window_sec": WINDOW_SECONDS,
                "fewshot_rank": rank,
                "split": side,
            }


def lowest_ranks(result, side, size):
    """Return {chunk_id: lowest rank of its identification examples} for the chunks of a split.

    Every test chunk is part of the test split, with rank 0. The examples of the unidentified
    vocalizations of the Sierra Nevada corpus count here like those of a taxon."""
    ranks = {}
    if side == "test":
        for chunk in result["chunks"]:
            if result["side"][chunk["rec_id"]] == "test":
                ranks[chunk["chunk_id"]] = 0
        return ranks
    for examples in result["examples"][side].values():
        for rank, (chunk_id, _) in enumerate(examples[:size]):
            if chunk_id not in ranks or rank < ranks[chunk_id]:
                ranks[chunk_id] = rank
    return ranks


def detection_rows(result, side, size):
    """Yield the detection chunks of one split, in the order of the recordings."""
    ranks = lowest_ranks(result, side, size)
    for chunk in result["chunks"]:
        if chunk["chunk_id"] not in ranks:
            continue
        samples, rate = read_chunk(chunk)
        events = []
        for event in chunk["events"]:
            events.append((float(event["onset"]), float(event["offset"])))
        events.sort()
        yield {
            "audio": {"path": f"{chunk['chunk_id']}.flac", "bytes": flac_bytes(samples, rate)},
            "sampling_rate": rate,
            "dataset": chunk["corpus"],
            "dataset_name": corpora.CORPUS_LABEL[chunk["corpus"]],
            "chunk_id": chunk["chunk_id"],
            "rec_id": chunk["rec_id"],
            "chunk_start": round(chunk["chunk_start"], 4),
            "duration": round(len(samples) / rate, 4),
            "n_events": len(events),
            "onsets": [round(onset, 4) for onset, _ in events],
            "offsets": [round(offset, 4) for _, offset in events],
            "fewshot_rank": ranks[chunk["chunk_id"]],
            "split": side,
        }


def write_part(part, columns, directory, name, index):
    """Write one parquet file of a split under a temporary name."""
    path = directory / f"{name}-{index:05d}.parquet.tmp"
    Dataset.from_list(part, features=columns).to_parquet(str(path))


def write_split(rows, columns, directory, name):
    """Write the rows of one split to parquet files of at most 1,500 rows or 400 MB of audio.

    Returns the number of rows."""
    part_sizes = []
    part = []
    audio_bytes = 0
    for row in rows:
        part.append(row)
        audio_bytes += len(row["audio"]["bytes"])
        if len(part) >= ROWS_PER_FILE or audio_bytes >= AUDIO_BYTES_PER_FILE:
            write_part(part, columns, directory, name, len(part_sizes))
            part_sizes.append(len(part))
            part = []
            audio_bytes = 0
    if part or not part_sizes:
        write_part(part, columns, directory, name, len(part_sizes))
        part_sizes.append(len(part))
    for index in range(len(part_sizes)):
        temporary = directory / f"{name}-{index:05d}.parquet.tmp"
        temporary.rename(directory / f"{name}-{index:05d}-of-{len(part_sizes):05d}.parquet")
    return sum(part_sizes)


def main():
    """Build both configurations of the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    result = build.build()
    scored = [taxon for taxon, entry in result["taxa"].items() if entry["scorable"]]
    print(f"{len(result['chunks'])} chunks, {len(result['taxa'])} taxa, {len(scored)} scored",
          flush=True)
    tasks = (("classification", IDENTIFICATION_COLUMNS, identification_rows),
             ("segmentation", DETECTION_COLUMNS, detection_rows))
    for task, columns, rows_of_split in tasks:
        directory = config.dataset_dir() / task
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        for name, side, size in SPLITS:
            written = write_split(rows_of_split(result, side, size), columns, directory, name)
            print(f"{task}/{name}: {written} rows", flush=True)


if __name__ == "__main__":
    main()
