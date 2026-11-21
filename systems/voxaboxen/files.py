"""Folders, file formats and the slow-down rule shared by the Voxaboxen scripts."""
import csv
import math

from benchmark import config

MODELS = ("detection", "taxon", "identification")
RUN_NAME = "run"

# Voxaboxen resamples every file to this rate.
MODEL_RATE = 16000

TABLE_HEADER = "Begin Time (s)\tEnd Time (s)\tAnnotation\n"
INFO_HEADER = "fn,audio_fp,selection_table_fp\n"


def input_dir(model, size):
    """Return the folder with the Voxaboxen input of one model and training size."""
    return config.DATA / "voxaboxen" / f"{model}_{size}"


def project_dir(model, size):
    """Return the Voxaboxen project folder of one model and training size."""
    return config.DATA / "models" / "voxaboxen" / f"{model}_{size}"


def run_dir(model, size):
    """Return the folder of the training run inside the project folder."""
    return project_dir(model, size) / RUN_NAME


def power_of_two_slow_down(rate):
    """Return the smallest power of two that brings `rate` down to the model rate or below."""
    return int(2 ** math.ceil(math.log2(rate / MODEL_RATE)))


def write_table(path, vocalizations):
    """Write (onset, offset, label) rows as a tab-separated selection table."""
    with open(path, "w") as handle:
        handle.write(TABLE_HEADER)
        for onset, offset, label in vocalizations:
            handle.write(f"{onset:.6f}\t{offset:.6f}\t{label}\n")


def write_info(path, rows):
    """Write (name, audio path, table path) rows as a Voxaboxen file list."""
    with open(path, "w") as handle:
        handle.write(INFO_HEADER)
        for name, audio_path, table_path in rows:
            handle.write(f"{name},{audio_path},{table_path}\n")


def read_predictions(directory, name, slow_down):
    """Return the combined forward and backward predictions of one file.

    The result is a list of (onset, offset, label) in the order of the table, with the times
    divided by the slow-down factor of the file. A file without a table has no predictions."""
    path = directory / f"peaks_pred_{name}-detthresh0.5-comb.txt"
    predictions = []
    if not path.exists():
        return predictions
    with open(path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            onset = float(row["Begin Time (s)"])
            offset = float(row["End Time (s)"])
            if offset > onset:
                label = (row["Annotation"] or "").strip()
                predictions.append((onset / slow_down, offset / slow_down, label))
    return predictions
