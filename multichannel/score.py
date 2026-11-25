"""Score the predictions of the three systems on the 32 test recordings.

A prediction matches a reference vocalization when both boundaries lie within 30 ms.

  detection    caller-independent F1: the per-bird predictions of a system are pooled, the
               caller of the reference is disregarded, matching is maximum one-to-one.
               Systems are compared across recordings with exact paired randomization tests.
  attribution  caller-specific precision, recall and F1 of the two systems with a focal
               sensor: predictions are matched to the reference of the same bird only, with
               greedy one-to-one matching. Paired bootstrap over recordings.
  overlap      detection of single reference vocalizations by overlap status, and of
               overlapping pairs. Paired bootstrap over vocalizations or pairs.
"""
import argparse
import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from data import COHORTS, SYSTEMS, data_root, in_cohort, test_recordings

TOLERANCE = 0.03
MIN_PAIR_OVERLAP = 0.02
REPLICATES = 2000
MIN_RECORDINGS = 6
COMPARISONS = (
    ("accelerometer", "microphone"),
    ("multichannel", "accelerometer"),
    ("multichannel", "microphone"),
)
FOCAL_SYSTEMS = ("accelerometer", "multichannel")
GROUP_COHORTS = ("four_birds", "eight_birds")


def load_predictions(root, system, test_set, name):
    """One list of (onset, offset) per channel."""
    path = root / "predictions" / system / test_set / f"{name}.json"
    data = json.loads(path.read_text())
    channels = []
    for channel in data["predictions"]:
        channels.append(list(zip(channel["onset"], channel["offset"])))
    return channels


def pooled_predictions(root, system, test_set, name):
    """The predictions of all channels of one recording in one list of (onset, offset)."""
    pooled = []
    for channel in load_predictions(root, system, test_set, name):
        pooled.extend(channel)
    return pooled


def pooled_reference(reference):
    """The reference vocalizations of all callers in one list of (onset, offset)."""
    intervals = []
    for calls in reference.values():
        intervals.extend(calls)
    return intervals


def maximum_matches(predictions, reference, tolerance=TOLERANCE):
    """Flag every reference interval matched in a maximum one-to-one matching."""
    predictions = np.asarray(predictions, dtype=float).reshape(-1, 2)
    reference = np.asarray(reference, dtype=float).reshape(-1, 2)
    matched = np.zeros(len(reference), dtype=bool)
    if len(predictions) == 0 or len(reference) == 0:
        return matched
    boundary_distance = np.abs(predictions[:, None, :] - reference[None, :, :])
    within_tolerance = (boundary_distance <= tolerance).all(axis=-1)
    positive_duration = predictions[:, 1] > predictions[:, 0]
    candidates = within_tolerance & positive_duration[:, None]
    # For every prediction the index of its matched reference interval, or -1.
    partner = maximum_bipartite_matching(csr_matrix(candidates), perm_type="column")
    matched[partner[partner >= 0]] = True
    return matched


def greedy_matches(predictions, reference):
    """Number of predictions matched, each taking the first free reference interval."""
    used = [False] * len(reference)
    for onset, offset in predictions:
        if offset <= onset:
            continue
        for i, (reference_onset, reference_offset) in enumerate(reference):
            onset_close = abs(onset - reference_onset) <= TOLERANCE
            offset_close = abs(offset - reference_offset) <= TOLERANCE
            if not used[i] and onset_close and offset_close:
                used[i] = True
                break
    return sum(used)


def scores(counts):
    """Precision, recall and F1 from (true positives, predictions, references), pooled over
    the second-to-last axis."""
    totals = np.asarray(counts, dtype=float).sum(axis=-2)
    true_positives = totals[..., 0]
    n_predictions = totals[..., 1]
    n_references = totals[..., 2]
    return {
        "precision": true_positives / np.maximum(n_predictions, 1),
        "recall": true_positives / np.maximum(n_references, 1),
        "f1": 2 * true_positives / np.maximum(n_predictions + n_references, 1),
    }


def bootstrap_p(differences):
    """Two-sided: twice the smaller fraction of replicates on either side of zero."""
    fraction_not_above = (differences <= 0).mean()
    fraction_not_below = (differences >= 0).mean()
    p_value = 2 * min(fraction_not_above, fraction_not_below)
    smallest_resolvable = 1 / len(differences)
    return float(min(1.0, max(p_value, smallest_resolvable)))


def exact_randomization_p(counts_a, counts_b):
    """Swap the two systems' counts within recordings in all 2^n ways; two-sided."""
    n_recordings = len(counts_a)
    observed = scores(counts_a)["f1"] - scores(counts_b)["f1"]
    # Bit i of the assignment number says whether recording i is swapped.
    assignments = np.arange(2 ** n_recordings)[:, None]
    swap = ((assignments >> np.arange(n_recordings)) & 1).astype(bool)[:, :, None]
    swapped_a = np.where(swap, counts_a, counts_b)
    swapped_b = np.where(swap, counts_b, counts_a)
    null = scores(swapped_a)["f1"] - scores(swapped_b)["f1"]
    return float(np.mean(np.abs(null) >= abs(observed) - 1e-12))


def holm(p_values):
    """Holm-adjusted P values of a {name: P value} family."""
    adjusted = {}
    running = 0.0
    for rank, key in enumerate(sorted(p_values, key=p_values.get)):
        remaining_tests = len(p_values) - rank
        running = max(running, min(1.0, remaining_tests * p_values[key]))
        adjusted[key] = running
    return adjusted


def recording_bootstrap(counts_a, counts_b, seed=42):
    """Differences a - b of precision, recall and F1 over resampled recordings."""
    n_recordings = len(counts_a)
    generator = np.random.default_rng(seed)
    index = generator.integers(n_recordings, size=(REPLICATES, n_recordings))
    resampled_a = scores(counts_a[index])
    resampled_b = scores(counts_b[index])
    differences = {}
    for metric in resampled_a:
        differences[metric] = resampled_a[metric] - resampled_b[metric]
    return differences


def cohort_mask(recordings, cohort):
    """One flag per test recording: whether it counts towards the cohort."""
    mask = []
    for recording_cohort, _, _, _ in recordings:
        mask.append(in_cohort(recording_cohort, cohort))
    return mask


def detection_counts(root, recordings):
    """Per system, one (true positives, predictions, references) row per recording, with
    the callers pooled."""
    counts = {}
    for system in SYSTEMS:
        counts[system] = []
    for _, test_set, name, reference in recordings:
        intervals = pooled_reference(reference)
        for system in SYSTEMS:
            pooled = pooled_predictions(root, system, test_set, name)
            true_positives = maximum_matches(pooled, intervals).sum()
            counts[system].append((true_positives, len(pooled), len(intervals)))
    return counts


def detection_test(counts_a, counts_b):
    """F1 difference a - b with its bootstrap interval and exact randomization P value."""
    difference = scores(counts_a)["f1"] - scores(counts_b)["f1"]
    bootstrap_differences = recording_bootstrap(counts_a, counts_b)["f1"]
    interval = np.quantile(bootstrap_differences, [0.025, 0.975])
    return {
        "difference": float(difference),
        "recording_bootstrap_95ci": interval.tolist(),
        "exact_p": exact_randomization_p(counts_a, counts_b),
    }


def detection(root, recordings):
    """Caller-independent F1 of every system per cohort, with the system comparisons."""
    counts = detection_counts(root, recordings)
    result = {}
    for cohort in COHORTS:
        keep = cohort_mask(recordings, cohort)
        rows = {}
        f1 = {}
        for system in SYSTEMS:
            rows[system] = np.array(counts[system])[keep]
            f1[system] = float(scores(rows[system])["f1"])
        tests = {}
        exact_p = {}
        for system_a, system_b in COMPARISONS:
            comparison = f"{system_a}_minus_{system_b}"
            tests[comparison] = detection_test(rows[system_a], rows[system_b])
            exact_p[comparison] = tests[comparison]["exact_p"]
        for comparison, adjusted_p in holm(exact_p).items():
            tests[comparison]["holm_p"] = adjusted_p
        result[cohort] = {"n_recordings": sum(keep), "f1": f1, "tests": tests}
    return result


def attribution_counts(root, recordings):
    """Per focal system, one (true positives, predictions, references) row per recording,
    with every bird's predictions matched to its own reference."""
    counts = {}
    for system in FOCAL_SYSTEMS:
        counts[system] = []
    for _, test_set, name, reference in recordings:
        for system in FOCAL_SYSTEMS:
            channels = load_predictions(root, system, test_set, name)
            true_positives = 0
            n_predictions = 0
            n_references = 0
            for bird, predictions in enumerate(channels):
                calls = reference.get(bird, [])
                true_positives += greedy_matches(predictions, calls)
                n_predictions += len(predictions)
                n_references += len(calls)
            counts[system].append((true_positives, n_predictions, n_references))
    return counts


def attribution(root, recordings):
    """Caller-specific precision, recall and F1 of the focal systems per cohort."""
    counts = attribution_counts(root, recordings)
    result = {}
    for cohort in COHORTS:
        keep = cohort_mask(recordings, cohort)
        result[cohort] = {"n_recordings": sum(keep)}
        rows = {}
        for system in FOCAL_SYSTEMS:
            rows[system] = np.array(counts[system])[keep]
            result[cohort][system] = {}
            for metric, value in scores(rows[system]).items():
                result[cohort][system][metric] = float(value)
        if sum(keep) >= MIN_RECORDINGS:
            differences = recording_bootstrap(rows["multichannel"], rows["accelerometer"])
            result[cohort]["bootstrap_p"] = {}
            for metric, difference in differences.items():
                result[cohort]["bootstrap_p"][metric] = bootstrap_p(difference)
    return result


def overlap_table(reference):
    """Per reference vocalization: (onset, offset), number of other callers with an
    intersecting vocalization and longest such intersection; and the pairs of
    vocalizations of different callers that intersect by more than the tolerance."""
    calls = []
    for caller, items in reference.items():
        for onset, offset in items:
            calls.append((caller, onset, offset))
    other_callers = [set() for _ in calls]
    longest = [0.0] * len(calls)
    pairs = []
    for i, (caller_i, onset_i, offset_i) in enumerate(calls):
        for k in range(i + 1, len(calls)):
            caller_k, onset_k, offset_k = calls[k]
            shared = min(offset_i, offset_k) - max(onset_i, onset_k)
            if caller_i == caller_k or shared <= 0:
                continue
            other_callers[i].add(caller_k)
            other_callers[k].add(caller_i)
            longest[i] = max(longest[i], shared)
            longest[k] = max(longest[k], shared)
            if shared > MIN_PAIR_OVERLAP:
                pairs.append((i, k))
    intervals = []
    n_others = []
    for (_, onset, offset), callers in zip(calls, other_callers):
        intervals.append((onset, offset))
        n_others.append(len(callers))
    return intervals, np.array(n_others), np.array(longest), pairs


def paired_bootstrap(hits, seed=3):
    """hits: (units, 3) detection flags for microphone, accelerometer, multichannel.
    P of multichannel against the other two for the detected fraction."""
    hits = np.asarray(hits, dtype=float).reshape(-1, len(SYSTEMS))
    generator = np.random.default_rng(seed)
    index = generator.integers(0, len(hits), (REPLICATES, len(hits)))
    fraction = hits[index].mean(axis=1)
    microphone = fraction[:, 0]
    accelerometer = fraction[:, 1]
    multichannel = fraction[:, 2]
    return {
        "vs_accelerometer": bootstrap_p(multichannel - accelerometer),
        "vs_microphone": bootstrap_p(multichannel - microphone),
        "n": len(hits),
    }


def paired_bootstrap_f1(hits, false_positives, seed=4):
    """Resample detection units: every reference vocalization (found or missed by each
    system) and every false-positive prediction (of one system); F1 per replicate."""
    hits = np.asarray(hits, dtype=float)
    # Columns of a unit: found by each system, false positive of each system, is a reference.
    reference_units = np.hstack([hits, np.zeros_like(hits), np.ones((len(hits), 1))])
    units = [reference_units]
    for column, count in enumerate(false_positives):
        false_positive_units = np.zeros((int(count), 7))
        false_positive_units[:, 3 + column] = 1
        units.append(false_positive_units)
    units = np.vstack(units)
    generator = np.random.default_rng(seed)
    index = generator.integers(0, len(units), (REPLICATES, len(units)))
    sums = units[index].sum(axis=1)
    true_positives = sums[:, :3]
    n_false_positives = sums[:, 3:6]
    n_references = sums[:, 6:7]
    f1 = 2 * true_positives / (true_positives + n_false_positives + n_references)
    microphone = f1[:, 0]
    accelerometer = f1[:, 1]
    multichannel = f1[:, 2]
    return {
        "vs_accelerometer": bootstrap_p(multichannel - accelerometer),
        "vs_microphone": bootstrap_p(multichannel - microphone),
        "n": len(units),
    }


def found(hits):
    """{system: fraction of the units it detected}."""
    hits = np.asarray(hits, dtype=float).reshape(-1, len(SYSTEMS))
    return dict(zip(SYSTEMS, hits.mean(axis=0).tolist()))


def found_with_test(hits):
    """Detected fraction per system and the paired bootstrap of multichannel against the
    other two systems."""
    return {"found": found(hits), "p": paired_bootstrap(hits)}


def overlap_units(root, recordings):
    """Tables over all test recordings. Per reference vocalization: cohort, detection flag
    of every system, number of other callers and longest overlap. Per overlapping pair:
    cohort and whether each system found both. Per cohort: false positives of every system."""
    cohorts = []
    hits = []
    n_others = []
    longest = []
    pair_cohorts = []
    pair_hits = []
    false_positives = {}
    for cohort, test_set, name, reference in recordings:
        intervals, others, shared, pairs = overlap_table(reference)
        if cohort not in false_positives:
            false_positives[cohort] = np.zeros(len(SYSTEMS))
        hit = np.zeros((len(intervals), len(SYSTEMS)), dtype=bool)
        for column, system in enumerate(SYSTEMS):
            pooled = pooled_predictions(root, system, test_set, name)
            hit[:, column] = maximum_matches(pooled, intervals)
            false_positives[cohort][column] += len(pooled) - hit[:, column].sum()
        cohorts += [cohort] * len(intervals)
        hits.append(hit)
        n_others.append(others)
        longest.append(shared)
        pair_cohorts += [cohort] * len(pairs)
        for first, second in pairs:
            pair_hits.append(hit[first] & hit[second])
    return {
        "cohorts": np.array(cohorts),
        "hits": np.vstack(hits),
        "n_others": np.concatenate(n_others),
        "longest": np.concatenate(longest),
        "pair_cohorts": np.array(pair_cohorts),
        "pair_hits": np.array(pair_hits),
        "false_positives": false_positives,
    }


def overlap(root, recordings):
    """Share of overlapping vocalizations and detection F1 tests per public cohort, and
    detection in the public groups by overlap status."""
    units = overlap_units(root, recordings)
    cohorts = units["cohorts"]
    hits = units["hits"]
    n_others = units["n_others"]
    longest = units["longest"]

    result = {"overlapping_fraction": {}, "f1_bootstrap_p": {}}
    for cohort in ("public_pairs", "four_birds", "eight_birds"):
        keep = cohorts == cohort
        cohort_false_positives = units["false_positives"][cohort]
        result["overlapping_fraction"][cohort] = float((n_others[keep] > 0).mean())
        result["f1_bootstrap_p"][cohort] = paired_bootstrap_f1(hits[keep], cohort_false_positives)

    groups = np.isin(cohorts, GROUP_COHORTS)
    subsets = {
        "all": groups,
        "alone": groups & (n_others == 0),
        "overlapping": groups & (n_others > 0),
        "one_other_bird": groups & (n_others == 1),
        "two_or_more_other_birds": groups & (n_others >= 2),
        "overlap_up_to_50ms": groups & (longest > 0) & (longest <= 0.05),
        "overlap_50_to_100ms": groups & (longest > 0.05) & (longest <= 0.1),
        "overlap_over_100ms": groups & (longest > 0.1),
    }
    result["public_groups"] = {}
    for key, keep in subsets.items():
        result["public_groups"][key] = found_with_test(hits[keep])
    group_pairs = np.isin(units["pair_cohorts"], GROUP_COHORTS)
    pair_hits = units["pair_hits"][group_pairs]
    result["public_groups"]["overlapping_pairs"] = found_with_test(pair_hits)
    return result


def percent_columns(fractions):
    """The fractions as percentages in columns of width 6."""
    columns = []
    for fraction in fractions:
        columns.append(f"{100 * fraction:6.1f}")
    return " ".join(columns)


def print_detection(result):
    """Print the caller-independent F1 per cohort."""
    print("Caller-independent F1 (%)        rec    MIC    ACC     MC   P MC-ACC  P MC-MIC")
    for cohort, row in result.items():
        f1 = [row["f1"][system] for system in SYSTEMS]
        p_accelerometer = row["tests"]["multichannel_minus_accelerometer"]["exact_p"]
        p_microphone = row["tests"]["multichannel_minus_microphone"]["exact_p"]
        line = f"  {cohort:28s} {row['n_recordings']:5d} {percent_columns(f1)}"
        print(f"{line} {p_accelerometer:9.3f} {p_microphone:9.3f}")


def print_attribution(result):
    """Print caller-specific precision, recall and F1 per cohort."""
    header = "Caller-specific (%)              rec  ACC: P      R     F1   MC: P      R     F1"
    print(f"{header}   P recall  P F1")
    for cohort, row in result.items():
        fractions = []
        for system in FOCAL_SYSTEMS:
            for metric in ("precision", "recall", "f1"):
                fractions.append(row[system][metric])
        line = f"  {cohort:28s} {row['n_recordings']:5d} {percent_columns(fractions)}"
        p_values = row.get("bootstrap_p")
        if p_values:
            line += f" {p_values['recall']:9.4f} {p_values['f1']:6.4f}"
        print(line)


def print_overlap(result):
    """Print the overlap analyses of the public recordings."""
    overlapping_percent = {}
    for cohort, fraction in result["overlapping_fraction"].items():
        overlapping_percent[cohort] = round(100 * fraction, 1)
    print("Vocalizations overlapping another bird (%):", overlapping_percent)
    print("Detection F1, P over vocalizations and false positives:")
    for cohort, p_values in result["f1_bootstrap_p"].items():
        p_accelerometer = p_values["vs_accelerometer"]
        p_microphone = p_values["vs_microphone"]
        print(f"  {cohort:28s} vs ACC {p_accelerometer:.4f}  vs MIC {p_microphone:.4f}")
    print("Found in public groups (%)         n    MIC    ACC     MC   P MC-ACC  P MC-MIC")
    for key, row in result["public_groups"].items():
        fractions = [row["found"][system] for system in SYSTEMS]
        p_values = row["p"]
        line = f"  {key:28s} {p_values['n']:5d} {percent_columns(fractions)}"
        print(f"{line} {p_values['vs_accelerometer']:9.4f} {p_values['vs_microphone']:9.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="data root (default: $CALLMARK_DATA)")
    args = parser.parse_args()
    root = data_root(args.data)
    recordings = test_recordings(root)
    result = {
        "detection": detection(root, recordings),
        "attribution": attribution(root, recordings),
        "overlap": overlap(root, recordings),
    }
    (root / "results").mkdir(exist_ok=True)
    (root / "results/scores.json").write_text(json.dumps(result, indent=1))

    print_detection(result["detection"])
    print_attribution(result["attribution"])
    print_overlap(result["overlap"])


if __name__ == "__main__":
    main()
