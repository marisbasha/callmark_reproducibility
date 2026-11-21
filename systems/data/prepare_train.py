"""Write the detection training windows: prepared/train_<size>/<chunk_id>__w<number>.wav and .json.

    python -m systems.data.prepare_train --data DIR

A training set of size N holds at most N annotated vocalizations of every taxon of every corpus.
Windows are cut from the train_500 chunks of the dataset:

- Pairs of taxon and corpus are processed rarest first. The identification examples of a pair
  are taken in the order of their rank until the pair holds N annotated vocalizations.
- Each example seeds a window around its vocalization. The window is widened until no
  annotation crosses its edges, then grows by one annotation at a time, nearest first, while it
  stays within 5 s and no pair in it exceeds N. A seed whose first window exceeds N is skipped.
- The window is extended over all vocalization-free audio around it, up to 5 ms from the next
  annotation or to the end of the chunk. Overlapping windows of a chunk are merged.

Unidentified vocalizations inside a window are annotated with the taxon "unknown" and are not
counted.
"""
import argparse
import math
from collections import Counter, defaultdict
from multiprocessing import Pool

from benchmark import config, data
from systems.data import chunks

# Annotations are added to a window while it stays within this length (seconds).
WINDOW = 5.0
# The audio around a window ends this far from the next annotation.
GAP = 0.005
# An annotation this close to the edge of a window belongs to the window.
EDGE = 0.001
# An annotation with less than this left inside the audio is not used.
MIN_LENGTH = 0.001


def lengths_of_file(dataset_file):
    """Return chunk_id -> audio length in seconds for the chunks of one dataset file."""
    lengths = {}
    for row in chunks.rows_with_audio(dataset_file):
        lengths[row["chunk_id"]] = chunks.audio_length(row)
    return lengths


def load_annotations(lengths):
    """Return chunk_id -> annotations of the train_500 chunks, offsets clipped to the audio.

    An annotation with nothing left inside the audio keeps its position with onset None."""
    taxa_of_chunk = chunks.annotation_taxa("train_500")
    rows = data.table("segmentation", "train_500", ("chunk_id", "onsets", "offsets"))
    annotations = {}
    for row in rows:
        length = lengths[row["chunk_id"]]
        onsets = []
        offsets = []
        for onset, offset in zip(row["onsets"], row["offsets"]):
            if offset > length:
                offset = length
            if offset - onset < MIN_LENGTH:
                onset = None
                offset = None
            onsets.append(onset)
            offsets.append(offset)
        annotations[row["chunk_id"]] = {
            "corpus": row["dataset"],
            "onsets": onsets,
            "offsets": offsets,
            "taxa": taxa_of_chunk[row["chunk_id"]],
            "length": length,
        }
    return annotations


def enclose(onsets, offsets, start, end):
    """Widen [start, end] until no annotation crosses an edge or lies within EDGE of it.

    Return the window and the indices of the annotations inside it."""
    while True:
        inside = []
        for index in range(len(onsets)):
            if onsets[index] is None:
                continue
            if onsets[index] < end + EDGE and offsets[index] > start - EDGE:
                inside.append(index)
        widened_start = min([start] + [onsets[index] for index in inside])
        widened_end = max([end] + [offsets[index] for index in inside])
        if widened_start == start and widened_end == end:
            return start, end, inside
        start = widened_start
        end = widened_end


def within_size(chunk_id, chunk, inside, counted, size):
    """Tell whether adding the annotations `inside` keeps every pair at or below `size`."""
    added = defaultdict(set)
    for index in inside:
        taxon = chunk["taxa"][index]
        if taxon is not None:
            added[(taxon, chunk["corpus"])].add((chunk_id, index))
    for pair, members in added.items():
        if len(counted[pair] | members) > size:
            return False
    return True


def distance_to(centre, onset, offset):
    """Return the distance of the centre of an annotation from `centre`."""
    return abs((onset + offset) / 2 - centre)


def grow_window(chunk_id, chunk, seed, counted, size):
    """Return (start, end, annotations inside) of the window seeded by annotation `seed`.

    Returns None when the first window around the seed exceeds the size of a pair."""
    onsets = chunk["onsets"]
    offsets = chunk["offsets"]
    start, end, inside = enclose(onsets, offsets, onsets[seed], offsets[seed])
    if not within_size(chunk_id, chunk, inside, counted, size):
        return None

    centre = (onsets[seed] + offsets[seed]) / 2
    usable = [index for index in range(len(onsets)) if onsets[index] is not None]
    distances = {}
    for index in usable:
        distances[index] = distance_to(centre, onsets[index], offsets[index])
    for index in sorted(usable, key=distances.get):
        if index in inside:
            continue
        wider_start = min(start, onsets[index])
        wider_end = max(end, offsets[index])
        wider_start, wider_end, wider_inside = enclose(onsets, offsets, wider_start, wider_end)
        if wider_end - wider_start > WINDOW:
            break
        if not within_size(chunk_id, chunk, wider_inside, counted, size):
            break
        start = wider_start
        end = wider_end
        inside = wider_inside

    quiet_start = 0.0
    quiet_end = chunk["length"]
    for index in usable:
        if offsets[index] <= start:
            quiet_start = max(quiet_start, offsets[index] + GAP)
        if onsets[index] >= end:
            quiet_end = min(quiet_end, onsets[index] - GAP)
    return min(quiet_start, start), max(quiet_end, end), inside


def fewshot_rank(example):
    """Return the rank of an identification example within its taxon."""
    return example["fewshot_rank"]


def select_windows(size, annotations):
    """Return chunk_id -> [(start, end)], the windows of the training set of one size."""
    examples = data.table(
        "classification", "train_500", ("species", "chunk_id", "event_index", "fewshot_rank")
    )
    examples_of_pair = defaultdict(list)
    for example in examples:
        pair = (example["species"], example["dataset"])
        examples_of_pair[pair].append(example)

    available = Counter()
    for chunk in annotations.values():
        for onset, taxon in zip(chunk["onsets"], chunk["taxa"]):
            if onset is not None and taxon is not None:
                available[(taxon, chunk["corpus"])] += 1

    # Rarest pair first; equally rare pairs in the order of their text form.
    rarity = {}
    for pair in examples_of_pair:
        rarity[pair] = (available[pair], str(pair))
    order = sorted(examples_of_pair, key=rarity.get)

    counted = defaultdict(set)
    windows = defaultdict(list)
    for pair in order:
        ranked = sorted(examples_of_pair[pair], key=fewshot_rank)
        for example in ranked:
            if len(counted[pair]) >= size:
                break
            chunk_id = example["chunk_id"]
            chunk = annotations[chunk_id]
            seed = example["event_index"]
            if chunk["taxa"][seed] != example["species"] or chunk["onsets"][seed] is None:
                continue
            if (chunk_id, seed) in counted[pair]:
                continue
            window = grow_window(chunk_id, chunk, seed, counted, size)
            if window is None:
                continue
            start, end, inside = window
            for index in inside:
                taxon = chunk["taxa"][index]
                if taxon is not None:
                    counted[(taxon, chunk["corpus"])].add((chunk_id, index))
            windows[chunk_id].append((start, end))
    return windows


def merge_windows(windows):
    """Return the windows of one chunk in order, overlapping windows merged."""
    ordered = sorted(windows)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def write_windows(directory, chunk_id, chunk, windows, waveform, sample_rate):
    """Cut the merged windows of one chunk and write each as a .wav and .json pair."""
    onsets = chunk["onsets"]
    offsets = chunk["offsets"]
    for number, (start, end) in enumerate(merge_windows(windows)):
        # Whole samples that cover the window.
        first_sample = int(math.floor(start * sample_rate))
        last_sample = min(len(waveform), int(math.ceil(end * sample_rate)))
        origin = first_sample / sample_rate
        duration = (last_sample - first_sample) / sample_rate
        labels = chunks.spectrogram_parameters(chunk["corpus"], sample_rate)
        labels["onset"] = []
        labels["offset"] = []
        labels["cluster"] = []
        labels["species"] = chunk["corpus"]
        for index in range(len(onsets)):
            if onsets[index] is None or onsets[index] < start or offsets[index] > end:
                continue
            labels["onset"].append(round(max(0.0, onsets[index] - origin), 4))
            labels["offset"].append(round(min(duration, offsets[index] - origin), 4))
            labels["cluster"].append(chunk["taxa"][index] or "unknown")
        name = f"{chunk_id}__w{number:02d}"
        audio = waveform[first_sample:last_sample]
        chunks.write_pair(directory, name, audio, sample_rate, labels)


def write_file(arguments):
    """Write the windows of every size that are cut from the chunks of one dataset file."""
    dataset_file, annotations, windows_of_size = arguments
    for row in chunks.rows_with_audio(dataset_file):
        chunk_id = row["chunk_id"]
        waveform = None
        for size, windows in windows_of_size.items():
            if chunk_id not in windows:
                continue
            if waveform is None:
                waveform, sample_rate = chunks.decode_audio(row)
            directory = config.prepared_dir(f"train_{size}")
            chunk = annotations[chunk_id]
            write_windows(directory, chunk_id, chunk, windows[chunk_id], waveform, sample_rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(config.SIZES))
    parser.add_argument("--workers", type=int, default=12, help="parallel processes")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    dataset_files = chunks.dataset_files("train_500")
    lengths = {}
    with Pool(arguments.workers) as pool:
        for lengths_of_one_file in pool.map(lengths_of_file, dataset_files):
            lengths.update(lengths_of_one_file)
    annotations = load_annotations(lengths)

    windows_of_size = {}
    for size in arguments.sizes:
        windows_of_size[size] = select_windows(size, annotations)
        config.prepared_dir(f"train_{size}").mkdir(parents=True, exist_ok=True)

    jobs = []
    for dataset_file in dataset_files:
        jobs.append((dataset_file, annotations, windows_of_size))
    with Pool(arguments.workers) as pool:
        pool.map(write_file, jobs)

    for size in arguments.sizes:
        directory = config.prepared_dir(f"train_{size}")
        count = len(list(directory.glob("*.json")))
        print(f"size {size}: {count} windows from {len(windows_of_size[size])} chunks")


if __name__ == "__main__":
    main()
