"""Score every system of the cross-taxon benchmark from its prediction files.

    python -m benchmark.score [--data DIR]

Writes results/benchmark.json under the data root and prints the tables of the paper.
"""
import argparse

import numpy as np

from benchmark import analyses, composition, config, data, metrics, probe, report, stats

DETECTORS = ("whisperseg", "whisperseg_no_pretraining", "voxaboxen", "tweetynet", "das")
END_TO_END = ("cascade", "dualwhisper", "voxaboxen", "birdcode")
IDENTIFIERS = ("whisperid", "birdcode", "dualwhisper", "biolingual", "naturelm", "voxaboxen")

# (system, comparator, training size) of the identification tests.
IDENTIFICATION_TESTS = [
    ("whisperid", "birdcode", 500),
    ("whisperid", "dualwhisper", 500),
    ("whisperid", "biolingual", 500),
    ("whisperid", "naturelm", 500),
    ("whisperid", "voxaboxen", 500),
    ("whisperid", "birdcode", 50),
    ("whisperid", "birdcode", 100),
    ("whisperid", "biolingual", 50),
    ("whisperid", "biolingual", 100),
]

# (encoder, comparator) of the tests between frozen probes at training size 500.
PROBE_TESTS = (
    ("whisperseg", "whisper_large"),
    ("whisperid_pretrained", "whisperseg"),
    ("whisperid_pretrained", "perch"),
)


# ---------------------------------------------------------------- detection
def reference_of(system):
    """Return the reference intervals of the test chunks as the detector read them.

    WhisperSeg read them from the prepared test set, the other detectors from the dataset."""
    if system.startswith("whisperseg"):
        return data.prepared_reference()
    return data.dataset_reference()


def f1_by_corpus(per_corpus):
    """Return {corpus: F1} from per-corpus scores."""
    return {corpus: scores["f1"] for corpus, scores in per_corpus.items()}


def score_detector(system, size):
    """Return all detection scores of one detector at one training size."""
    predicted = data.predictions(system, "detection", size)
    reference = reference_of(system)
    per_corpus = metrics.detection(data.chunks(predicted, reference), secondary=True)

    common_chunks = data.chunks(predicted, data.prepared_reference())
    common = metrics.detection(common_chunks)

    nine_chunks = data.chunks(predicted, reference, config.BIRDCODE_CORPORA)
    nine = metrics.detection(nine_chunks, config.FRAME_TOLERANCE)

    restored_chunks = data.chunks(predicted, analyses.restore_long(reference))
    restored = metrics.detection(restored_chunks)

    merged = analyses.merged_detection(predicted)
    merged_f1 = []
    merged_f1_others = []
    for corpus, scores in merged.items():
        merged_f1.append(scores["f1_merged"])
        if corpus not in analyses.CONTIGUOUS:
            merged_f1_others.append(scores["f1_merged"])

    out = {}
    for key in ("f1", "f1_maximum", "iou_f1", "frame_f1"):
        out[key] = metrics.macro(per_corpus, key)
    out["per_corpus"] = per_corpus
    # Precision and recall are scored against the common (prepared) reference.
    out["precision"] = metrics.macro(common, "precision")
    out["recall"] = metrics.macro(common, "recall")
    out["f1_frame_tolerance"] = metrics.macro(nine)
    out["f1_restored"] = metrics.macro(restored)
    out["f1_merged"] = float(np.mean(merged_f1))
    out["f1_merged_others"] = float(np.mean(merged_f1_others))
    out["merged_per_corpus"] = merged
    return out


def detection():
    """Return the detection scores of every detector, the baselines and the Wilcoxon tests."""
    out = {}
    for system in DETECTORS:
        out[system] = {}
        for size in config.SIZES:
            out[system][size] = score_detector(system, size)

    energy_predicted = data.predictions("energy", "detection")
    energy = metrics.detection(data.chunks(energy_predicted, data.dataset_reference()))
    out["energy"] = {"f1": metrics.macro(energy), "per_corpus": energy}

    birdcode_predicted = data.predictions("birdcode", "detection", 500)
    birdcode_chunks = data.chunks(
        birdcode_predicted, data.dataset_reference(), config.BIRDCODE_CORPORA
    )
    birdcode = metrics.detection(birdcode_chunks, config.FRAME_TOLERANCE)
    out["birdcode"] = {
        500: {"f1_frame_tolerance": metrics.macro(birdcode), "per_corpus": birdcode}
    }

    whisperseg_f1 = f1_by_corpus(out["whisperseg"][500]["per_corpus"])
    out["wilcoxon_500"] = {}
    for other in ("voxaboxen", "tweetynet", "das"):
        other_f1 = f1_by_corpus(out[other][500]["per_corpus"])
        out["wilcoxon_500"][other] = stats.wilcoxon_corpora(whisperseg_f1, other_f1)
    return out


# ---------------------------------------------------------------- end-to-end
def summarize_end_to_end(per_corpus):
    """Return the corpus means of the end-to-end scores, with the per-corpus scores."""
    matched = 0
    for scores in per_corpus.values():
        matched += scores["matched"]
    return {
        "f1": metrics.macro(per_corpus),
        "detection_f1": metrics.macro(per_corpus, "detection_f1"),
        "conditional_accuracy": metrics.macro(per_corpus, "conditional_accuracy"),
        "matched": matched,
        "per_corpus": per_corpus,
    }


def score_end_to_end(predicted, reference, restored, reported_at_frame_tolerance):
    """Return the end-to-end scores of one set of predictions.

    `reported_at_frame_tolerance` says that the system is reported at the frame tolerance over the
    nine corpora, which sets the basis of the score with restored references."""
    eleven_chunks = data.chunks(predicted, reference)
    at_corpus_tolerance = metrics.end_to_end(eleven_chunks)

    nine_chunks = data.chunks(predicted, reference, config.BIRDCODE_CORPORA)
    at_frame_tolerance = metrics.end_to_end(nine_chunks, config.FRAME_TOLERANCE)

    if reported_at_frame_tolerance:
        restored_chunks = data.chunks(predicted, restored, config.BIRDCODE_CORPORA)
        with_restored = metrics.end_to_end(restored_chunks, config.FRAME_TOLERANCE)
    else:
        restored_chunks = data.chunks(predicted, restored, config.CORPORA)
        with_restored = metrics.end_to_end(restored_chunks, None)

    return {
        "corpus_tolerance": summarize_end_to_end(at_corpus_tolerance),
        "frame_tolerance": summarize_end_to_end(at_frame_tolerance),
        "f1_restored": metrics.macro(with_restored),
    }


def end_to_end():
    """Return the end-to-end scores on the Detection+Identification subset.

    They are given at the corpus tolerances over the 11 corpora, and at the frame tolerance over
    the nine corpora within BirdCODE's input range; then with the vocalizations removed by the
    duration filter restored, on the basis each system is reported on."""
    reference = data.end_to_end_reference()
    restored = analyses.restore_long(reference)
    out = {}
    for system in END_TO_END:
        out[system] = {}
        for size in config.SIZES:
            predicted = data.predictions(system, "end_to_end", size)
            out[system][size] = score_end_to_end(
                predicted, reference, restored, system == "birdcode"
            )

    cascade_f1 = f1_by_corpus(out["cascade"][500]["corpus_tolerance"]["per_corpus"])
    voxaboxen_f1 = f1_by_corpus(out["voxaboxen"][500]["corpus_tolerance"]["per_corpus"])
    birdcode_f1 = f1_by_corpus(out["birdcode"][500]["frame_tolerance"]["per_corpus"])
    out["wilcoxon_500"] = {
        "voxaboxen": stats.wilcoxon_corpora(cascade_f1, voxaboxen_f1),
        "birdcode": stats.wilcoxon_corpora(cascade_f1, birdcode_f1),
    }
    return out


# ---------------------------------------------------------------- identification
def predicted_taxa(system, size):
    """Return the predicted taxon of every identification test example as an array."""
    return np.array(data.identification_predictions(system, size))


def macro_f1_by_size(system, truth, taxa, examples):
    """Return {training size: macro-F1} of one system on the selected test examples."""
    scores = {}
    for size in config.SIZES:
        predicted = predicted_taxa(system, size)[examples]
        scores[size] = metrics.macro_f1(truth[examples], predicted, taxa)
    return scores


def compare(system_a, system_b, size, truth, taxa, examples):
    """Return the paired bootstrap of the macro-F1 difference a - b on the selected examples."""
    predicted_a = predicted_taxa(system_a, size)[examples]
    predicted_b = predicted_taxa(system_b, size)[examples]
    return stats.bootstrap_macro_f1(truth[examples], predicted_a, predicted_b, taxa)


def identification():
    """Return the macro-F1 over the scored taxa and the tests between systems.

    Fine-tuned systems are scored on the test examples of the 11 corpora, the frozen-encoder
    probes on all test examples of the dataset."""
    _, truth, _ = data.identification_test()
    truth = np.array(truth)
    taxa = data.scored_taxa()
    scored = np.array(composition.scored_examples())
    everything = np.ones(len(truth), dtype=bool)

    out = {"macro_f1": {}, "difference": {}, "probe_difference_500": {}}
    for system in IDENTIFIERS:
        out["macro_f1"][system] = macro_f1_by_size(system, truth, taxa, scored)
    for encoder in probe.ENCODERS:
        system = f"probe_{encoder}"
        out["macro_f1"][system] = macro_f1_by_size(system, truth, taxa, everything)
    for system, comparator, size in IDENTIFICATION_TESTS:
        test = compare(system, comparator, size, truth, taxa, scored)
        out["difference"].setdefault(comparator, {})[size] = test
    for encoder, comparator in PROBE_TESTS:
        test = compare(f"probe_{encoder}", f"probe_{comparator}", 500, truth, taxa, everything)
        out["probe_difference_500"][f"{encoder} - {comparator}"] = test
    return out


# ---------------------------------------------------------------- overlap and tolerance
def overlap_and_tolerance():
    """Return the overlap analysis and the tolerance sweep, at training size 500.

    The sweep gives the recall of short vocalizations as a function of tolerance."""
    predicted = {}
    for system in ("whisperseg", "voxaboxen", "tweetynet", "das", "birdcode"):
        predicted[system] = data.predictions(system, "detection", 500)

    short = {}
    for system in ("whisperseg", "voxaboxen", "birdcode"):
        short[system] = analyses.short_recall(predicted[system])
    corpora, counts, whisperseg_hits = short["whisperseg"]

    sweep = {"tolerances": analyses.SWEEP, "recall": {}, "vocalizations": {}, "p": {}}
    for system in short:
        _, _, hits = short[system]
        if system == "birdcode":
            keep = np.isin(corpora, config.BIRDCODE_CORPORA)
        else:
            keep = np.isin(corpora, config.CORPORA)
        n_vocalizations = counts[keep].sum()
        sweep["recall"][system] = (hits[keep].sum(axis=0) / n_vocalizations).tolist()
        sweep["vocalizations"][system] = int(n_vocalizations)
        if system != "whisperseg":
            sweep["p"][system] = stats.bootstrap_chunks(
                whisperseg_hits[keep], hits[keep], counts[keep]
            )
    return analyses.overlap(predicted), sweep


def main():
    """Score everything, write the result file and print the tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()
    config.use_data_root(args.data)
    results = {}
    results["detection"] = detection()
    results["end_to_end"] = end_to_end()
    results["identification"] = identification()
    results["overlap"], results["short_recall"] = overlap_and_tolerance()
    results["evaluation_sets"] = composition.evaluation_sets()
    results["training_sets"] = composition.training_sets()
    data.write_json(config.results_dir() / "benchmark.json", results)
    report.print_tables(results)


if __name__ == "__main__":
    main()
