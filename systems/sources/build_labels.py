"""Write the two label files that need the annotations of the original corpora.

    python -m systems.sources.build_labels --data DIR

labels/excluded_long_test.json: {chunk_id: [[onset, offset, taxon], ...]}, the vocalizations
removed by the duration filter that overlap a test chunk, clipped to the chunk.

labels/train_500_source_annotations.json: {chunk_id: [[onset, offset], ...]}, every annotation
of the original recording that overlaps a train_500 chunk, not clipped. It includes the
vocalizations that the duration filter removed.

Times are seconds from the start of the chunk. Reads corpora/ and the chunk identifiers and
chunk starts of dataset/segmentation.
"""
import argparse
from collections import defaultdict

from benchmark import config, data
from systems.sources import build, corpora

# Annotations up to this distance outside a training chunk are listed (s).
MARGIN = 0.001


def excluded_long_test(recordings):
    """Return the removed long vocalizations of the test chunks."""
    chunks, longest = build.build_chunks(recordings)
    test_chunk_ids = set(data.dataset_reference())
    chunks_of_recording = defaultdict(list)
    for chunk in chunks:
        if chunk["chunk_id"] in test_chunk_ids:
            chunks_of_recording[chunk["rec_id"]].append(chunk)
    excluded = defaultdict(list)
    for item in recordings:
        for onset, offset, taxon in item["events"]:
            if offset - onset <= longest[item["corpus"]]:
                continue
            for chunk in chunks_of_recording.get(item["rec_id"], []):
                start = chunk["chunk_start"]
                end = chunk["chunk_end"]
                if onset < end and offset > start:
                    clipped = [max(onset, start) - start, min(offset, end) - start, taxon]
                    excluded[chunk["chunk_id"]].append(clipped)
    return excluded


def train_source_annotations(recordings):
    """Return the annotations of the original recordings around every train_500 chunk."""
    events_of_recording = defaultdict(list)
    for item in recordings:
        for onset, offset, _ in item["events"]:
            events_of_recording[item["rec_id"]].append((onset, offset))
    columns = ("chunk_id", "rec_id", "chunk_start", "duration")
    annotations = {}
    for row in data.table("segmentation", "train_500", columns):
        start = row["chunk_start"]
        inside = []
        for onset, offset in events_of_recording[row["rec_id"]]:
            if onset - start < row["duration"] + MARGIN and offset - start > -MARGIN:
                inside.append([onset - start, offset - start])
        annotations[row["chunk_id"]] = inside
    return annotations


def main():
    """Write both label files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    recordings = corpora.read_all()
    excluded = excluded_long_test(recordings)
    data.write_json(config.label_file("excluded_long_test.json"), excluded)
    print(f"{sum(len(events) for events in excluded.values())} removed vocalizations in "
          f"{len(excluded)} test chunks")
    annotations = train_source_annotations(recordings)
    data.write_json(config.label_file("train_500_source_annotations.json"), annotations)
    print(f"{sum(len(events) for events in annotations.values())} annotations around "
          f"{len(annotations)} training chunks")


if __name__ == "__main__":
    main()
