"""Constants and file locations of the cross-taxon benchmark.

Everything is read from and written to one data root, given by the environment variable
CALLMARK_DATA or by use_data_root() (the --data argument of every script). README.md describes
its layout.
"""
import os
from pathlib import Path

# Training sizes: at most this many vocalizations per taxon.
SIZES = (50, 100, 500)

# The 11 scored corpora. Right whale and spider recordings are part of the training material only;
# rodents2019 duplicates the recordings of rodent_usv and is dropped wherever the dataset is read.
CORPORA = (
    "budgie",
    "canary",
    "enabirds",
    "infant_marmosets",
    "nbm",
    "ozf",
    "rodent_usv",
    "singing_mice",
    "sne_full",
    "tworadiobirds",
    "wytham",
)
ULTRASONIC = ("rodent_usv", "singing_mice")
DUPLICATE_CORPUS = "rodents2019"

# This taxon is in the label vocabulary but is not scored.
EXCLUDED_TAXON = "Rattus norvegicus"

CORPUS_NAME = {
    "budgie": "Budgerigar",
    "canary": "Domestic canary",
    "enabirds": "Eastern North American birds",
    "infant_marmosets": "Infant marmosets",
    "nbm": "NBM nocturnal birds",
    "ozf": "Zebra finch group",
    "rodent_usv": "Rodent USV",
    "singing_mice": "Singing mice",
    "sne_full": "Sierra Nevada soundscape",
    "tworadiobirds": "Zebra finch pairs",
    "wytham": "Wytham great tit",
}

# Onset and offset tolerance in seconds:
# min(50 ms, max(10 ms, 0.2 x median duration of the corpus's test vocalizations)).
TOLERANCE = {
    "budgie": 0.0130,
    "canary": 0.0500,
    "enabirds": 0.0500,
    "infant_marmosets": 0.0225,
    "nbm": 0.0305,
    "ozf": 0.0214,
    "rodent_usv": 0.0100,
    "singing_mice": 0.0101,
    "sne_full": 0.0500,
    "tworadiobirds": 0.0144,
    "wytham": 0.0186,
}

# One BirdCODE output frame (131.6 ms).
FRAME_TOLERANCE = 1 / 7.6

# The corpora within BirdCODE's 32-kHz input.
BIRDCODE_CORPORA = tuple(corpus for corpus in CORPORA if corpus not in ULTRASONIC)


def frame_length(corpus):
    """Return the frame length in seconds used for the binary frame F1 of a corpus."""
    if corpus in ULTRASONIC:
        return 0.0002
    return 0.001


FRAME = {corpus: frame_length(corpus) for corpus in CORPORA}

BOOTSTRAP_REPLICATES = 2000

DATA = Path(os.environ.get("CALLMARK_DATA", "data"))


def use_data_root(path):
    """Set the data root to `path`, unless `path` is empty."""
    global DATA
    if path:
        DATA = Path(path)


def dataset_dir():
    """Return the directory of the published dataset."""
    return DATA / "dataset"


def prepared_dir(name):
    """Return the directory of prepared detection audio and labels.

    `name` is 'test', 'train_50', 'train_100' or 'train_500'."""
    return DATA / "prepared" / name


def label_file(name):
    """Return the path of a label file."""
    return DATA / "labels" / name


def embedding_file(encoder):
    """Return the path of the embeddings of one encoder."""
    return DATA / "embeddings" / f"{encoder}.npy"


def prediction_file(system, task, size=None):
    """Return predictions/<system>/<task>_<size>.json.gz (identification: .json).

    Systems without training omit the size."""
    if size is None:
        stem = task
    else:
        stem = f"{task}_{size}"
    if task == "identification":
        extension = ".json"
    else:
        extension = ".json.gz"
    return DATA / "predictions" / system / (stem + extension)


def results_dir():
    """Return the directory of the result file."""
    return DATA / "results"
