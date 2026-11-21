"""Benchmark endpoints.

Detection: a predicted vocalization matches a reference vocalization when both its onset and its
offset lie within the corpus tolerance. Predictions are matched greedily in prediction order, each
reference at most once. Precision, recall and F1 are computed within each corpus; the score is the
unweighted mean over corpora (corpus-macro).

Detection+Identification: the same matching with the taxon required to agree (class-aware
event-based F-measure, Mesaros et al. 2016).

Identification: per-taxon F1 averaged over the scored taxa (taxon-macro).
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from benchmark import config


def intervals(vocalizations):
    """Return (onsets, offsets) as arrays from a list of (onset, offset, ...) tuples."""
    onsets = np.array([vocalization[0] for vocalization in vocalizations], dtype=float)
    offsets = np.array([vocalization[1] for vocalization in vocalizations], dtype=float)
    return onsets, offsets


def within(pred, ref, tolerance, inclusive=False):
    """Return the boolean matrix [prediction, reference] of pairs with both boundaries close.

    Both boundaries must differ by less than the tolerance, or by at most the tolerance when
    `inclusive` is set."""
    pred_onsets, pred_offsets = intervals(pred)
    ref_onsets, ref_offsets = intervals(ref)
    onset_error = np.abs(pred_onsets[:, None] - ref_onsets[None, :])
    offset_error = np.abs(pred_offsets[:, None] - ref_offsets[None, :])
    if inclusive:
        return (onset_error <= tolerance) & (offset_error <= tolerance)
    return (onset_error < tolerance) & (offset_error < tolerance)


def greedy(allowed):
    """Return the matched reference index of every prediction (-1: unmatched).

    The matching is one-to-one: each prediction, in order, takes the first free reference it is
    allowed to match."""
    free = np.ones(allowed.shape[1], dtype=bool)
    matched_reference = np.full(allowed.shape[0], -1)
    for prediction in np.flatnonzero(allowed.any(axis=1)):
        candidates = np.flatnonzero(allowed[prediction] & free)
        if len(candidates):
            matched_reference[prediction] = candidates[0]
            free[candidates[0]] = False
    return matched_reference


def match(pred, ref, tolerance):
    """Return the matched reference index of every prediction under the two-boundary tolerance."""
    return greedy(within(pred, ref, tolerance))


def count_matched(matched_reference):
    """Return the number of matched predictions in the output of greedy()."""
    return int((matched_reference >= 0).sum())


def maximum(allowed):
    """Return the size of the maximum one-to-one matching.

    `allowed` is a boolean matrix [prediction, reference]."""
    rows, columns = linear_sum_assignment(-allowed.astype(float))
    return int(allowed[rows, columns].sum())


def iou(pred, ref):
    """Return the matrix [prediction, reference] of temporal intersection over union."""
    pred_onsets, pred_offsets = intervals(pred)
    ref_onsets, ref_offsets = intervals(ref)
    latest_onset = np.maximum(pred_onsets[:, None], ref_onsets[None, :])
    earliest_onset = np.minimum(pred_onsets[:, None], ref_onsets[None, :])
    latest_offset = np.maximum(pred_offsets[:, None], ref_offsets[None, :])
    earliest_offset = np.minimum(pred_offsets[:, None], ref_offsets[None, :])
    intersection = earliest_offset - latest_onset
    union = latest_offset - earliest_onset
    defined = union > 0
    safe_union = np.where(defined, union, 1)
    return np.where(defined, np.maximum(intersection, 0) / safe_union, 0)


def frame_mask(vocalizations, frame, n_frames):
    """Return the boolean mask of the frames covered by a list of (onset, offset) intervals."""
    mask = np.zeros(n_frames, dtype=bool)
    for onset, offset in vocalizations:
        first = max(0, int(onset / frame))
        end = min(n_frames, int(np.ceil(offset / frame)))
        mask[first:end] = True
    return mask


def count_frames(pred, ref, frame):
    """Return (common, predicted, reference) vocal frames after rasterising both interval lists."""
    pred_offsets = [vocalization[1] for vocalization in pred]
    ref_offsets = [vocalization[1] for vocalization in ref]
    last_offset = max(pred_offsets + ref_offsets + [0.0])
    n_frames = max(1, int(np.ceil(last_offset / frame)))
    pred_mask = frame_mask(pred, frame, n_frames)
    ref_mask = frame_mask(ref, frame, n_frames)
    return int((pred_mask & ref_mask).sum()), int(pred_mask.sum()), int(ref_mask.sum())


def prf(true_positives, n_pred, n_ref):
    """Return precision, recall and F1, rounded to four decimals as tabulated."""
    if n_pred:
        precision = true_positives / n_pred
    else:
        precision = 0.0
    if n_ref:
        recall = true_positives / n_ref
    else:
        recall = 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def macro(per_corpus, key="f1"):
    """Return the unweighted mean over corpora of one value.

    Corpora where the value is undefined (None) are skipped."""
    values = []
    for scores in per_corpus.values():
        if scores[key] is not None:
            values.append(scores[key])
    return sum(values) / len(values)


def detection(chunks, tolerance=None, secondary=False):
    """Return per-corpus detection precision, recall and F1.

    `chunks` is [(corpus, predicted, reference)]. `tolerance` replaces the corpus tolerances by
    one value. With `secondary`, also IoU-F1 (maximum one-to-one matching at IoU >= 0.5), binary
    frame F1, and the detection F1 under maximum instead of greedy matching."""
    counts = defaultdict(Counter)
    for corpus, pred, ref in chunks:
        tally = counts[corpus]
        tally["pred"] += len(pred)
        tally["ref"] += len(ref)
        allowed = within(pred, ref, tolerance or config.TOLERANCE[corpus])
        tally["tp"] += count_matched(greedy(allowed))
        if secondary:
            tally["tp_maximum"] += maximum(allowed)
            tally["iou"] += maximum(iou(pred, ref) >= 0.5)
            frames = count_frames(pred, ref, config.FRAME[corpus])
            common_frames, pred_frames, ref_frames = frames
            tally["frame_tp"] += common_frames
            tally["frame_pred"] += pred_frames
            tally["frame_ref"] += ref_frames
    per_corpus = {}
    for corpus in sorted(counts):
        tally = counts[corpus]
        scores = prf(tally["tp"], tally["pred"], tally["ref"])
        if secondary:
            with_maximum = prf(tally["tp_maximum"], tally["pred"], tally["ref"])
            with_iou = prf(tally["iou"], tally["pred"], tally["ref"])
            with_frames = prf(tally["frame_tp"], tally["frame_pred"], tally["frame_ref"])
            scores["f1_maximum"] = with_maximum["f1"]
            scores["iou_f1"] = with_iou["f1"]
            scores["frame_f1"] = with_frames["f1"]
        per_corpus[corpus] = scores
    return per_corpus


def same_taxon_matrix(pred, ref):
    """Return the boolean matrix [prediction, reference] of pairs with equal taxon."""
    same_taxon = np.zeros((len(pred), len(ref)), dtype=bool)
    for i, predicted_vocalization in enumerate(pred):
        for k, reference_vocalization in enumerate(ref):
            same_taxon[i, k] = predicted_vocalization[2] == reference_vocalization[2]
    return same_taxon


def end_to_end(chunks, tolerance=None):
    """Return per-corpus end-to-end scores.

    `chunks` is [(corpus, predicted, reference)] with (onset, offset, taxon) entries. A boundary
    exactly at the tolerance counts as within it.

    f1                    taxon and both boundaries correct; unidentified reference vocalizations
                          (taxon None) are left out of the references, every prediction stays in
                          the precision denominator
    detection_f1          boundaries only, all reference vocalizations
    conditional_accuracy  fraction of boundary-matched identified reference vocalizations with the
                          correct taxon
    """
    counts = defaultdict(Counter)
    for corpus, pred, ref in chunks:
        tally = counts[corpus]
        identified = []
        for index, reference_vocalization in enumerate(ref):
            if reference_vocalization[2] is not None:
                identified.append(index)
        tally["pred"] += len(pred)
        tally["ref"] += len(ref)
        tally["identified"] += len(identified)
        if not pred or not ref:
            continue
        near = within(pred, ref, tolerance or config.TOLERANCE[corpus], inclusive=True)
        same_taxon = same_taxon_matrix(pred, ref)
        near_identified = near[:, identified]
        same_taxon_identified = same_taxon[:, identified]
        tally["found"] += count_matched(greedy(near))
        tally["correct"] += count_matched(greedy(near_identified & same_taxon_identified))
        # Conditional accuracy: match on boundaries alone, then check the taxon of each pair.
        matched_reference = greedy(near_identified)
        is_matched = matched_reference >= 0
        pair_has_same_taxon = same_taxon_identified[is_matched, matched_reference[is_matched]]
        tally["matched"] += count_matched(matched_reference)
        tally["named"] += int(pair_has_same_taxon.sum())
    per_corpus = {}
    for corpus in sorted(counts):
        tally = counts[corpus]
        if tally["matched"]:
            conditional_accuracy = round(tally["named"] / tally["matched"], 4)
        else:
            conditional_accuracy = None
        per_corpus[corpus] = {
            "f1": prf(tally["correct"], tally["pred"], tally["identified"])["f1"],
            "detection_f1": prf(tally["found"], tally["pred"], tally["ref"])["f1"],
            "conditional_accuracy": conditional_accuracy,
            "matched": tally["matched"],
        }
    return per_corpus


def taxon_counts(truth, pred, taxa):
    """Return the array [example, taxon, 3] of indicators.

    The three indicators are true positive, false positive and false negative."""
    index = {taxon: position for position, taxon in enumerate(taxa)}
    counts = np.zeros((len(truth), len(taxa), 3), dtype=np.int32)
    for row, (true_taxon, predicted_taxon) in enumerate(zip(truth, pred)):
        if true_taxon == predicted_taxon:
            if true_taxon in index:
                counts[row, index[true_taxon], 0] = 1
        else:
            if predicted_taxon in index:
                counts[row, index[predicted_taxon], 1] = 1
            if true_taxon in index:
                counts[row, index[true_taxon], 2] = 1
    return counts


def taxon_f1(counts):
    """Return the per-taxon F1 from [..., taxon, 3] counts.

    The three counts are true positives, false positives and false negatives."""
    true_positives = counts[..., 0]
    false_positives = counts[..., 1]
    false_negatives = counts[..., 2]
    denominator = 2 * true_positives + false_positives + false_negatives
    f1 = np.zeros(denominator.shape)
    np.divide(2 * true_positives, denominator, out=f1, where=denominator > 0)
    return f1


def macro_f1(truth, pred, taxa):
    """Return the identification macro-F1 over `taxa`.

    Every example enters the confusion counts: a prediction of a scored taxon on an example of an
    unscored taxon is a false positive."""
    totals = taxon_counts(truth, pred, taxa).sum(axis=0)
    return float(taxon_f1(totals).mean())
