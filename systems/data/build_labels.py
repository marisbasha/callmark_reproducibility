"""Write labels/test_taxa.json and labels/end_to_end_chunks.json from the dataset.

    python -m systems.data.build_labels --data DIR

test_taxa.json holds the taxon of every annotation of every test chunk. end_to_end_chunks.json
lists the Detection+Identification subset: at most 300 test chunks per corpus, taken at an even
stride through the chunk identifiers in sorted order.
"""
import argparse
from collections import defaultdict

from benchmark import config, data
from systems.data import chunks

CHUNKS_PER_CORPUS = 300


def end_to_end_subset(chunk_ids):
    """Return the chunks of the subset, corpus by corpus in sorted order."""
    chunks_of_corpus = defaultdict(list)
    for chunk_id in chunk_ids:
        chunks_of_corpus[data.corpus_of(chunk_id)].append(chunk_id)
    subset = []
    for corpus in sorted(chunks_of_corpus):
        ordered = sorted(chunks_of_corpus[corpus])
        if len(ordered) <= CHUNKS_PER_CORPUS:
            subset.extend(ordered)
        else:
            stride = len(ordered) / CHUNKS_PER_CORPUS
            for position in range(CHUNKS_PER_CORPUS):
                subset.append(ordered[int(position * stride)])
    return subset


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    taxa = chunks.annotation_taxa("test")
    data.write_json(config.label_file("test_taxa.json"), taxa)
    subset = end_to_end_subset(list(taxa))
    data.write_json(config.label_file("end_to_end_chunks.json"), subset)
    print(f"{len(taxa)} test chunks, {len(subset)} in the Detection+Identification subset")


if __name__ == "__main__":
    main()
