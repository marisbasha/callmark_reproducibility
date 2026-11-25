"""Score the two manual annotations of the public recordings against the consolidated reference.

The consolidated reference was built from a transmitter-based annotation, made on the
accelerometer channels, and a microphone-based annotation. Both are scored like a detector:
caller-independent, both boundaries within 30 ms, maximum one-to-one matching.
"""
import argparse
import json

import numpy as np

from data import PUBLIC_COHORT, data_root, in_cohort, public_info, public_recordings, \
    public_reference, public_segments
from score import maximum_matches, overlap_table, scores

ANNOTATIONS = ("transmitter-based", "microphone-based")
COHORTS = ("public_pairs", "four_birds", "eight_birds", "public_groups")


def annotated_intervals(root, path, kind):
    """The (onset, offset) of every vocalization of one manual annotation of a recording."""
    intervals = []
    for _, onset, offset in public_segments(root, path, kind):
        intervals.append((onset, offset))
    return intervals


def score_recording(root, path):
    """Cohort of a public recording, overlap flag of every reference vocalization and, per
    annotation, which reference vocalizations and overlapping pairs it found."""
    bird_names, _ = public_info(path)
    intervals, n_others, _, pairs = overlap_table(public_reference(root, path))
    row = {"cohort": PUBLIC_COHORT[len(bird_names)], "overlapping": n_others > 0}
    for kind in ANNOTATIONS:
        annotated = annotated_intervals(root, path, kind)
        hit = maximum_matches(annotated, intervals)
        both_found = []
        for first, second in pairs:
            both_found.append(hit[first] and hit[second])
        row[kind] = {"hit": hit, "n": len(annotated), "pairs": both_found}
    return row


def summarize(rows, kind):
    """F1 of one annotation over the given recordings and the share of reference
    vocalizations it found: all, alone, overlapping, and both of an overlapping pair."""
    overlapping = []
    hits = []
    pairs = []
    counts = []
    for row in rows:
        hit = row[kind]["hit"]
        overlapping.append(row["overlapping"])
        hits.append(hit)
        pairs.extend(row[kind]["pairs"])
        counts.append((hit.sum(), row[kind]["n"], len(hit)))
    overlapping = np.concatenate(overlapping)
    hits = np.concatenate(hits)
    return {
        "f1": float(scores(counts)["f1"]),
        "found": float(hits.mean()),
        "found_alone": float(hits[~overlapping].mean()),
        "found_overlapping": float(hits[overlapping].mean()),
        "found_overlapping_pairs": float(np.mean(pairs)),
    }


def print_table(result):
    """Print one line per cohort and annotation, in percent."""
    print("Manual annotation against the consolidated reference (%)")
    print("  cohort          annotation            F1  found  alone  overlapping  pairs")
    for cohort, row in result.items():
        for kind, values in row.items():
            f1 = 100 * values["f1"]
            found = 100 * values["found"]
            alone = 100 * values["found_alone"]
            overlapping = 100 * values["found_overlapping"]
            pairs = 100 * values["found_overlapping_pairs"]
            line = f"  {cohort:15s} {kind:18s} {f1:6.1f} {found:6.1f} {alone:6.1f}"
            print(f"{line} {overlapping:12.1f} {pairs:6.1f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="data root (default: $CALLMARK_DATA)")
    args = parser.parse_args()
    root = data_root(args.data)

    rows = []
    for path, _ in public_recordings(root):
        rows.append(score_recording(root, path))

    result = {}
    for cohort in COHORTS:
        selected = []
        for row in rows:
            if in_cohort(row["cohort"], cohort):
                selected.append(row)
        result[cohort] = {}
        for kind in ANNOTATIONS:
            result[cohort][kind] = summarize(selected, kind)
    (root / "results").mkdir(exist_ok=True)
    (root / "results/annotators.json").write_text(json.dumps(result, indent=1))
    print_table(result)


if __name__ == "__main__":
    main()
