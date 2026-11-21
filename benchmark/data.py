"""Access to the benchmark dataset, the reference annotations and the prediction files."""
import functools
import gzip
import json

import pyarrow.parquet as pq

from benchmark import config


def read_json(path):
    """Return the content of a JSON file; a name ending in .gz is read as gzip."""
    path = str(path)
    if path.endswith(".gz"):
        handle = gzip.open(path, "rt")
    else:
        handle = open(path)
    with handle:
        return json.load(handle)


def write_json(path, value):
    """Write `value` as JSON, creating the directory; a .gz suffix is written as gzip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        handle = gzip.open(path, "wt")
    else:
        handle = open(path, "w")
    with handle:
        json.dump(value, handle)


def corpus_of(chunk_id):
    """Return the corpus name, the part of a chunk identifier before the first '__'."""
    return chunk_id.split("__")[0]


@functools.lru_cache(maxsize=None)
def table(task, split, columns):
    """Return the rows of one split of the dataset as dictionaries.

    task: 'classification' (identification examples) or 'segmentation' (detection chunks), the
    names of the two dataset configurations; split: 'test', 'train_50', 'train_100' or
    'train_500'. The audio column is read only when asked for."""
    files = sorted((config.dataset_dir() / task).glob(f"{split}-*.parquet"))
    file_names = [str(file) for file in files]
    dataset = pq.ParquetDataset(file_names)
    rows = dataset.read(columns=list(columns) + ["dataset"]).to_pylist()
    kept = []
    for row in rows:
        if row["dataset"] != config.DUPLICATE_CORPUS:
            kept.append(row)
    return kept


# ---------------------------------------------------------------- identification
def example_id(row):
    """Return the identifier of an identification example: <chunk_id>__e<event index>."""
    return f"{row['chunk_id']}__e{row['event_index']:04d}"


def identification_test():
    """Return (ids, taxa, corpora) of the identification test examples, in dataset order."""
    rows = table("classification", "test", ("chunk_id", "event_index", "species"))
    ids = [example_id(row) for row in rows]
    taxa = [row["species"] for row in rows]
    corpora = [row["dataset"] for row in rows]
    return ids, taxa, corpora


def scored_taxa():
    """Return the sorted taxa that enter macro-F1.

    These are the taxa meeting the support criteria in one of the scored corpora."""
    rows = table("classification", "test", ("species", "scorable"))
    taxa = set()
    for row in rows:
        if row["scorable"] and row["dataset"] in config.CORPORA:
            taxa.add(row["species"])
    return sorted(taxa - {config.EXCLUDED_TAXON})


def identification_predictions(system, size):
    """Return the predicted taxon of every identification test example, in dataset order."""
    predicted = read_json(config.prediction_file(system, "identification", size))
    ids, _, _ = identification_test()
    return [predicted[example] for example in ids]


# ---------------------------------------------------------------- detection and end-to-end
@functools.lru_cache(maxsize=None)
def dataset_reference():
    """Return chunk_id -> [(onset, offset)], the test annotations as stored in the dataset."""
    rows = table("segmentation", "test", ("chunk_id", "onsets", "offsets"))
    reference = {}
    for row in rows:
        reference[row["chunk_id"]] = list(zip(row["onsets"], row["offsets"]))
    return reference


@functools.lru_cache(maxsize=None)
def prepared_reference():
    """Return chunk_id -> [(onset, offset)] as written to the prepared detection test set.

    These are the annotations of the dataset with times rounded to 0.1 ms and offsets clipped to
    the end of the audio."""
    reference = {}
    for path in sorted(config.prepared_dir("test").glob("*.json")):
        labels = read_json(path)
        reference[path.stem] = list(zip(labels["onset"], labels["offset"]))
    return reference


@functools.lru_cache(maxsize=None)
def test_taxa():
    """Return chunk_id -> taxon of every reference vocalization, in annotation order.

    The taxon of an unidentified vocalization is None."""
    return read_json(config.label_file("test_taxa.json"))


def end_to_end_reference():
    """Return chunk_id -> [(onset, offset, taxon)] for the Detection+Identification subset."""
    reference = dataset_reference()
    taxa = test_taxa()
    chunk_ids = read_json(config.label_file("end_to_end_chunks.json"))
    labelled = {}
    for chunk_id in chunk_ids:
        vocalizations = []
        for (onset, offset), taxon in zip(reference[chunk_id], taxa[chunk_id]):
            vocalizations.append((onset, offset, taxon))
        labelled[chunk_id] = vocalizations
    return labelled


# ---------------------------------------------------------------- predictions
def predictions(system, task, size=None):
    """Return chunk_id -> predicted vocalizations of one system.

    A vocalization is (onset, offset) for detection and (onset, offset, taxon) for end_to_end."""
    rows = read_json(config.prediction_file(system, task, size))
    predicted = {}
    for row in rows:
        predicted[row["chunk_id"]] = [tuple(vocalization) for vocalization in row["pred"]]
    return predicted


def chunks(predicted, reference, corpora=config.CORPORA):
    """Return [(corpus, predicted, reference)] for every reference chunk of `corpora`.

    A chunk without a prediction entry is an error."""
    paired = []
    for chunk_id, reference_vocalizations in reference.items():
        corpus = corpus_of(chunk_id)
        if corpus in corpora:
            paired.append((corpus, predicted[chunk_id], reference_vocalizations))
    return paired
