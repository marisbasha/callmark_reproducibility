"""Composition of the benchmark: evaluation sets, training-set sizes and training material per
scored taxon."""
from collections import Counter

import numpy as np
import soundfile

from benchmark import config, data


def scored_examples():
    """Return a mask over the identification test examples.

    It selects the examples of the scored corpora, without the excluded taxon."""
    _, taxa, corpora = data.identification_test()
    mask = []
    for taxon, corpus in zip(taxa, corpora):
        mask.append(corpus in config.CORPORA and taxon != config.EXCLUDED_TAXON)
    return mask


def scored_corpora_only(reference):
    """Return the entries of {chunk_id: vocalizations} that belong to the scored corpora."""
    kept = {}
    for chunk_id, vocalizations in reference.items():
        if data.corpus_of(chunk_id) in config.CORPORA:
            kept[chunk_id] = vocalizations
    return kept


def count_vocalizations(reference):
    """Return the total number of vocalizations in {chunk_id: vocalizations}."""
    total = 0
    for vocalizations in reference.values():
        total += len(vocalizations)
    return total


def identification_set():
    """Return the size of the identification test set of the scored corpora."""
    rows = data.table("classification", "test", ("species", "rec_id"))
    scored = set(data.scored_taxa())
    examples = 0
    unscored = 0
    recordings = set()
    for row, keep in zip(rows, scored_examples()):
        if not keep:
            continue
        examples += 1
        if row["species"] not in scored:
            unscored += 1
        recordings.add(row["rec_id"])
    return {
        "examples": examples,
        "unscored": unscored,
        "recordings": len(recordings),
        "scored_taxa": len(scored),
    }


def detection_set():
    """Return the size of the detection test set of the scored corpora."""
    reference = scored_corpora_only(data.dataset_reference())
    rows = data.table("segmentation", "test", ("chunk_id", "rec_id"))
    recordings = set()
    for row in rows:
        if row["chunk_id"] in reference:
            recordings.add(row["rec_id"])
    return {
        "chunks": len(reference),
        "vocalizations": count_vocalizations(reference),
        "recordings": len(recordings),
    }


def end_to_end_set():
    """Return the size of the Detection+Identification subset of the scored corpora."""
    reference = scored_corpora_only(data.end_to_end_reference())
    unidentified = 0
    for vocalizations in reference.values():
        for _, _, taxon in vocalizations:
            if taxon is None:
                unidentified += 1
    return {
        "chunks": len(reference),
        "vocalizations": count_vocalizations(reference),
        "unidentified": unidentified,
    }


def evaluation_sets():
    """Return the sizes of the three evaluation sets."""
    return {
        "identification": identification_set(),
        "detection": detection_set(),
        "end_to_end": end_to_end_set(),
    }


def distribution(values, size):
    """Return the range and quartiles of `values`, and how many of them equal `size`."""
    first_quartile, median, third_quartile = np.percentile(values, [25, 50, 75])
    at_cap = 0
    for value in values:
        if value == size:
            at_cap += 1
    return {
        "min": int(min(values)),
        "q1": float(first_quartile),
        "median": float(median),
        "q3": float(third_quartile),
        "max": int(max(values)),
        "at_cap": at_cap,
    }


def detection_training_material(windows, scored):
    """Return what the detection training windows hold.

    The result is (seconds of audio, vocalizations per scored taxon and corpus, annotated minutes
    per scored taxon)."""
    seconds = 0.0
    vocalizations = Counter()
    minutes = Counter()
    for path in windows:
        info = soundfile.info(str(path.with_suffix(".wav")))
        seconds += info.frames / info.samplerate
        labels = data.read_json(path)
        corpus = data.corpus_of(path.stem)
        for onset, offset, taxon in zip(labels["onset"], labels["offset"], labels["cluster"]):
            if taxon in scored and corpus in config.CORPORA:
                vocalizations[taxon, corpus] += 1
                minutes[taxon] += (offset - onset) / 60
    return seconds, vocalizations, minutes


def training_sets():
    """Return the composition of the training set of every training size.

    Per size: identification examples; detection windows and their audio hours; identification
    examples per scored taxon; annotated vocalizations, and their summed duration in minutes, per
    scored taxon and corpus in the detection windows."""
    scored = set(data.scored_taxa())
    composition = {}
    for size in config.SIZES:
        examples = data.table("classification", f"train_{size}", ("species",))
        examples_per_taxon = Counter(row["species"] for row in examples)
        windows = sorted(config.prepared_dir(f"train_{size}").glob("*.json"))
        seconds, vocalizations, minutes = detection_training_material(windows, scored)
        examples_of_scored = [examples_per_taxon[taxon] for taxon in scored]
        minutes_of_scored = [minutes[taxon] for taxon in scored]
        vocalization_counts = list(vocalizations.values())
        composition[size] = {
            "identification_examples": len(examples),
            "detection_windows": len(windows),
            "detection_window_hours": seconds / 3600,
            "identification_examples_per_taxon": distribution(examples_of_scored, size),
            "detection_vocalizations_per_taxon_and_corpus": distribution(
                vocalization_counts, size
            ),
            "detection_minutes_per_taxon_median": float(np.median(minutes_of_scored)),
        }
    return composition
