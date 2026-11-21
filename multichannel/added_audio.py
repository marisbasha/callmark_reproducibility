"""Microphone-only detection on the 14 public group recordings (four and eight birds) as
a function of the training audio, with the other systems and the two manual annotations.

Scored are the three systems trained on the 9.4 h of pair recordings, the microphone-only
detector trained with 7.3, 29.9 and 91.2 h of added group audio labelled by MC-WhisperSeg
(added_audio_sets.py), and the transmitter-based and microphone-based manual annotations.
For each, against the consolidated reference with the callers pooled:

  f1               both boundaries within 20 ms, maximum one-to-one matching
  frame_f1         1-ms frames, vocal or not
  found            share of reference vocalizations found when 1, 2 or 3 and more birds
                   vocalize at the same time
  f1_by_tolerance  f1 with both boundaries within 5 to 200 ms

Writes results/added_audio.json.
"""
import argparse
import json

import numpy as np

from data import data_root, public_info, public_recordings, public_reference, public_segments
from score import load_predictions, maximum_matches, overlap_table

PAIR_HOURS = 9.4
ANNOTATIONS = ("transmitter-based", "microphone-based")
PAIR_TRAINED_SYSTEMS = ("multichannel", "accelerometer", "microphone")
ADDED_HOURS = {"7h": 7.3, "30h": 29.9, "91h": 91.2}
TOLERANCES = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
FRAME = 0.001
SIMULTANEOUS = ("1", "2", "3+")


def frame_mask(intervals, n_frames):
    """One flag per frame: whether it lies in one of the intervals."""
    mask = np.zeros(n_frames, dtype=bool)
    for onset, offset in intervals:
        first_frame = max(0, int(onset / FRAME))
        end_frame = min(n_frames, int(np.ceil(offset / FRAME)))
        mask[first_frame:end_frame] = True
    return mask


def frame_counts(predictions, reference):
    """(common, predicted, reference) vocal frames."""
    last_offset = FRAME
    for _, offset in predictions + reference:
        last_offset = max(last_offset, offset)
    n_frames = int(np.ceil(last_offset / FRAME))
    predicted_frames = frame_mask(predictions, n_frames)
    reference_frames = frame_mask(reference, n_frames)
    common_frames = predicted_frames & reference_frames
    return common_frames.sum(), predicted_frames.sum(), reference_frames.sum()


def f1(counts):
    """F1 from (common, predicted, reference) counts summed over recordings."""
    common, predicted, reference = np.sum(counts, axis=0)
    return float(2 * common / (predicted + reference))


def annotated_intervals(root, path, kind):
    """The (onset, offset) of every vocalization of one manual annotation of a recording."""
    intervals = []
    for _, onset, offset in public_segments(root, path, kind):
        intervals.append((onset, offset))
    return intervals


def predicted_intervals(model_root, system, path):
    """The (onset, offset) predicted by a system for a recording, all channels pooled."""
    intervals = []
    for channel in load_predictions(model_root, system, "test_public", path.stem):
        intervals.extend(channel)
    return intervals


def all_sources(root, recordings):
    """{name: (training hours, {recording: intervals})} of the manual annotations, the
    systems trained on pairs and the microphone detectors trained with added audio."""
    sources = {}
    for kind in ANNOTATIONS:
        intervals = {}
        for path in recordings:
            intervals[path] = annotated_intervals(root, path, kind)
        sources[f"{kind} annotation"] = (None, intervals)
    for system in PAIR_TRAINED_SYSTEMS:
        intervals = {}
        for path in recordings:
            intervals[path] = predicted_intervals(root, system, path)
        sources[system] = (PAIR_HOURS, intervals)
    for set_name, added_hours in ADDED_HOURS.items():
        model_root = root.parent / "added_audio" / set_name / "multichannel"
        intervals = {}
        for path in recordings:
            intervals[path] = predicted_intervals(model_root, "microphone", path)
        sources[f"microphone, {set_name} added"] = (PAIR_HOURS + added_hours, intervals)
    return sources


def evaluate(intervals, tables):
    """Scores of one source. intervals: {recording: its (onset, offset) list}; tables:
    {recording: (reference intervals, number of other birds vocalizing with each)}."""
    segment_counts = []
    frames = []
    hits = []
    n_others = []
    sweep = {}
    for tolerance in TOLERANCES:
        sweep[tolerance] = []
    for path, (reference, simultaneous) in tables.items():
        predictions = intervals[path]
        hit = maximum_matches(predictions, reference)
        segment_counts.append((hit.sum(), len(predictions), len(reference)))
        frames.append(frame_counts(predictions, reference))
        hits.append(hit)
        n_others.append(simultaneous)
        for tolerance in TOLERANCES:
            n_matched = maximum_matches(predictions, reference, tolerance).sum()
            sweep[tolerance].append((n_matched, len(predictions), len(reference)))
    hits = np.concatenate(hits)
    n_others = np.concatenate(n_others)
    selected = {"1": n_others == 0, "2": n_others == 1, "3+": n_others >= 2}
    found = {}
    n_vocalizations = {}
    for n_birds in SIMULTANEOUS:
        found[n_birds] = float(hits[selected[n_birds]].mean())
        n_vocalizations[n_birds] = int(selected[n_birds].sum())
    f1_by_tolerance = {}
    for tolerance, counts in sweep.items():
        f1_by_tolerance[f"{1000 * tolerance:g} ms"] = f1(counts)
    return {
        "f1": f1(segment_counts),
        "frame_f1": f1(frames),
        "found": found,
        "n_vocalizations": n_vocalizations,
        "f1_by_tolerance": f1_by_tolerance,
    }


def print_scores(result):
    """Print F1, frame F1 and the share found by number of simultaneous birds, in percent."""
    header = "14 group recordings (%)          hours     F1  frame F1"
    print(f"{header}   found: 1 bird  2 birds  3+ birds")
    for name, row in result.items():
        hours = "     "
        if row["training_hours"]:
            hours = f"{row['training_hours']:5.1f}"
        f1_percent = 100 * row["f1"]
        frame_f1_percent = 100 * row["frame_f1"]
        found_1 = 100 * row["found"]["1"]
        found_2 = 100 * row["found"]["2"]
        found_3 = 100 * row["found"]["3+"]
        line = f"  {name:30s} {hours} {f1_percent:6.1f} {frame_f1_percent:9.1f}"
        print(f"{line} {found_1:14.0f} {found_2:8.0f} {found_3:8.0f}")


def print_tolerance_sweep(result):
    """Print F1 in percent for every boundary tolerance."""
    tolerance_columns = []
    for tolerance in TOLERANCES:
        tolerance_columns.append(f"{1000 * tolerance:5g}")
    print("F1 (%) by boundary tolerance      " + " ".join(tolerance_columns) + " ms")
    for name, row in result.items():
        f1_columns = []
        for value in row["f1_by_tolerance"].values():
            f1_columns.append(f"{100 * value:5.1f}")
        print(f"  {name:30s}  " + " ".join(f1_columns))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="data root (default: $CALLMARK_DATA)")
    args = parser.parse_args()
    root = data_root(args.data)

    tables = {}
    for path, _ in public_recordings(root):
        bird_names, _ = public_info(path)
        if len(bird_names) > 2:
            reference, n_others, _, _ = overlap_table(public_reference(root, path))
            tables[path] = (reference, n_others)

    result = {}
    for name, (hours, intervals) in all_sources(root, list(tables)).items():
        result[name] = {"training_hours": hours, **evaluate(intervals, tables)}
    (root / "results").mkdir(exist_ok=True)
    (root / "results/added_audio.json").write_text(json.dumps(result, indent=1))

    print_scores(result)
    print_tolerance_sweep(result)


if __name__ == "__main__":
    main()
