"""Supplementary analyses of the detection results: merging of adjacent vocalizations, restoring
the vocalizations removed by the duration filter, temporal overlap, and recall of short
vocalizations as a function of tolerance. All of them rescore the stored predictions; nothing is
retrained."""
from collections import Counter, defaultdict

import numpy as np

from benchmark import config, data, metrics, stats

# Corpora whose references tile vocal sequences with back-to-back units.
CONTIGUOUS = ("infant_marmosets", "canary")

# Longest duration, in seconds, of a short vocalization.
SHORT = 0.2

# Common tolerances of the sweep, in seconds.
SWEEP = (0.01, 0.02, 0.04, 0.08, 0.1)


def f1(true_positives, n_pred, n_ref):
    """Return the unrounded F1 from the counts of matches, predictions and references."""
    if n_pred + n_ref:
        return 2 * true_positives / (n_pred + n_ref)
    return 0.0


# ---------------------------------------------------------------- merging adjacent vocalizations
def merge(vocalizations, tolerance, keys=None):
    """Return the sorted (onset, offset) intervals after merging adjacent vocalizations.

    Consecutive vocalizations of the same key are merged when they are separated by a gap of 0 to
    `tolerance` seconds. Overlapping vocalizations are never merged; a merged vocalization can
    absorb the next one."""
    if not keys:
        keys = [None] * len(vocalizations)
    streams = defaultdict(list)
    for vocalization, key in zip(vocalizations, keys):
        streams[key].append(list(vocalization[:2]))
    merged = []
    for stream in streams.values():
        stream.sort()
        current = stream[0]
        for onset, offset in stream[1:]:
            gap = onset - current[1]
            if 0 <= gap <= tolerance:
                current[1] = max(current[1], offset)
            else:
                merged.append(tuple(current))
                current = [onset, offset]
        merged.append(tuple(current))
    return sorted(merged)


def merged_detection(predicted):
    """Return the detection F1 per corpus before and after merging.

    References are merged within a taxon and predictions regardless of label; predictions are
    matched in onset order."""
    reference = data.prepared_reference()
    taxa = data.test_taxa()
    counts = defaultdict(Counter)
    for chunk_id, ref in reference.items():
        corpus = data.corpus_of(chunk_id)
        if corpus not in config.CORPORA:
            continue
        tolerance = config.TOLERANCE[corpus]
        tally = counts[corpus]
        pred = sorted(predicted[chunk_id])
        merged_pred = merge(pred, tolerance)
        merged_ref = merge(ref, tolerance, taxa[chunk_id])
        matched = metrics.match(pred, ref, tolerance)
        matched_after_merging = metrics.match(merged_pred, merged_ref, tolerance)
        tally["tp"] += metrics.count_matched(matched)
        tally["tp_merged"] += metrics.count_matched(matched_after_merging)
        tally["pred"] += len(pred)
        tally["ref"] += len(ref)
        tally["pred_merged"] += len(merged_pred)
        tally["ref_merged"] += len(merged_ref)
    per_corpus = {}
    for corpus in sorted(counts):
        tally = counts[corpus]
        per_corpus[corpus] = {
            "f1": f1(tally["tp"], tally["pred"], tally["ref"]),
            "f1_merged": f1(tally["tp_merged"], tally["pred_merged"], tally["ref_merged"]),
            "references": tally["ref"],
            "references_merged": tally["ref_merged"],
        }
    return per_corpus


# ---------------------------------------------------------------- duration filter
def first_vocalization(reference):
    """Return the first vocalization found in {chunk_id: vocalizations}."""
    for vocalizations in reference.values():
        if vocalizations:
            return vocalizations[0]


def boundaries(vocalization):
    """Return the (onset, offset) part of a vocalization."""
    return vocalization[:2]


def restore_long(reference):
    """Return the references with the vocalizations removed by the duration filter put back.

    The restored vocalizations are clipped to their chunk and have as many fields as the others."""
    excluded = data.read_json(config.label_file("excluded_long_test.json"))
    n_fields = len(first_vocalization(reference))
    restored = {}
    for chunk_id, ref in reference.items():
        vocalizations = list(ref)
        for vocalization in excluded.get(chunk_id, []):
            vocalizations.append(tuple(vocalization[:n_fields]))
        restored[chunk_id] = sorted(vocalizations, key=boundaries)
    return restored


# ---------------------------------------------------------------- temporal overlap
def intersections(ref):
    """Return the matrix of pairwise intersection lengths of the reference intervals.

    A negative length means that the two intervals are disjoint."""
    onsets, offsets = metrics.intervals(ref)
    earliest_offset = np.minimum(offsets[:, None], offsets[None, :])
    latest_onset = np.maximum(onsets[:, None], onsets[None, :])
    return earliest_offset - latest_onset


def largest_boundary_shift(ref, original):
    """Return the largest absolute difference between two versions of the same boundaries."""
    return np.abs(np.array(ref) - np.array(original)).max(initial=0)


def overlap(predicted, frame_systems=("birdcode",)):
    """Return recall by overlap stratum per system, and each system's difference to the first.

    Recall is given on all, isolated and overlapping reference vocalizations, together with the
    fraction of overlapping pairs (overlap longer than the corpus tolerance) with both members
    matched, pooled over vocalizations; then the difference between the first system and every
    other one, by paired bootstrap within corpora.

    predicted: {system: {chunk_id: predictions}}. Systems in `frame_systems` are matched at the
    frame tolerance and scored on the corpora within their input range. Chunks whose reference
    boundaries differ by more than 50 ms between the dataset and the prepared test set are left
    out."""
    reference = data.prepared_reference()
    original = data.dataset_reference()
    corpus_of_vocalization = []
    is_overlapping = []
    corpus_of_pair = []
    hit = {system: [] for system in predicted}
    resolved = {system: [] for system in predicted}
    for chunk_id, ref in reference.items():
        corpus = data.corpus_of(chunk_id)
        if corpus not in config.CORPORA:
            continue
        if largest_boundary_shift(ref, original[chunk_id]) >= 0.05:
            continue
        tolerance = config.TOLERANCE[corpus]
        shared = intersections(ref)
        pairs = np.argwhere(np.triu(shared > tolerance, k=1))
        # A vocalization overlaps when it shares time with another one, not with itself.
        np.fill_diagonal(shared, 0)
        corpus_of_vocalization += [corpus] * len(ref)
        is_overlapping += (shared > 0).any(axis=1).tolist()
        corpus_of_pair += [corpus] * len(pairs)
        for system in predicted:
            if system in frame_systems:
                system_tolerance = config.FRAME_TOLERANCE
            else:
                system_tolerance = tolerance
            found = metrics.match(predicted[system][chunk_id], ref, system_tolerance)
            matched = np.isin(np.arange(len(ref)), found)
            both_matched = matched[pairs[:, 0]] & matched[pairs[:, 1]]
            hit[system] += matched.tolist()
            resolved[system] += both_matched.tolist()
    corpus_of_vocalization = np.array(corpus_of_vocalization)
    is_overlapping = np.array(is_overlapping)
    corpus_of_pair = np.array(corpus_of_pair)

    everything = np.ones(len(corpus_of_vocalization), dtype=bool)
    strata = {"all": everything, "isolated": ~is_overlapping, "overlapping": is_overlapping}
    in_range = {}
    for system in predicted:
        if system in frame_systems:
            in_range[system] = np.isin(corpus_of_vocalization, config.BIRDCODE_CORPORA)
        else:
            in_range[system] = everything

    vocalizations = {}
    for name, stratum in strata.items():
        vocalizations[name] = int(stratum.sum())
    out = {
        "vocalizations": vocalizations,
        "pairs": len(corpus_of_pair),
        "recall": {},
        "difference": {},
    }
    for system in predicted:
        hit[system] = np.array(hit[system])
        resolved[system] = np.array(resolved[system])
        recall = {}
        for name, stratum in strata.items():
            recall[name] = float(hit[system][stratum & in_range[system]].mean())
        recall["pairs"] = float(resolved[system].mean())
        out["recall"][system] = recall

    first, *others = predicted
    rng = np.random.default_rng(0)
    for system in others:
        pair_test = stats.bootstrap_within_corpora(
            resolved[first], resolved[system], corpus_of_pair, rng
        )
        out["difference"][system] = {"pairs": pair_test}
    for name, stratum in strata.items():
        for system in others:
            keep = stratum & in_range[system]
            out["difference"][system][name] = stats.bootstrap_within_corpora(
                hit[first][keep], hit[system][keep], corpus_of_vocalization[keep], rng
            )
    return out


# ---------------------------------------------------------------- short vocalizations, tolerance
def short_recall(predicted):
    """Return (corpus [chunk], counts [chunk], hits [chunk, tolerance]) for short vocalizations.

    Per test chunk of the scored corpora: the number of reference vocalizations up to 200 ms long,
    and how many of them are matched at each common tolerance of SWEEP (greedy one-to-one matching
    against all references of the chunk)."""
    corpora = []
    counts = []
    hits = []
    for chunk_id, ref in data.prepared_reference().items():
        corpus = data.corpus_of(chunk_id)
        if corpus not in config.CORPORA:
            continue
        is_short = [offset - onset <= SHORT for onset, offset in ref]
        short = np.flatnonzero(is_short)
        hits_by_tolerance = []
        for tolerance in SWEEP:
            matched = metrics.match(predicted[chunk_id], ref, tolerance)
            hits_by_tolerance.append(len(np.intersect1d(matched, short)))
        corpora.append(corpus)
        counts.append(len(short))
        hits.append(hits_by_tolerance)
    return np.array(corpora), np.array(counts, dtype=float), np.array(hits, dtype=float)
