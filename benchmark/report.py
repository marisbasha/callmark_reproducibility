"""Print the benchmark tables from the results of score.py."""
from benchmark import analyses, config

NAMES = {
    "whisperseg": "WhisperSeg",
    "whisperseg_no_pretraining": "WhisperSeg without detection pretraining",
    "whisperid": "WhisperID",
    "cascade": "Cascade (WhisperSeg + WhisperID)",
    "dualwhisper": "DualWhisper",
    "voxaboxen": "Voxaboxen",
    "tweetynet": "TweetyNet",
    "das": "DAS",
    "birdcode": "BirdCODE",
    "biolingual": "BioLingual",
    "naturelm": "NatureLM-audio",
    "energy": "Energy baseline",
    "probe_whisperseg": "released WhisperSeg",
    "probe_whisper_large": "Whisper large",
    "probe_whisperid_bridge": "bridge stage",
    "probe_whisperid_bridge_xcl1": "bridge stage + one XCL epoch",
    "probe_whisperid_pretrained": "bridge stage + two XCL epochs (WhisperID)",
    "probe_xeno_canto_only": "xeno-canto only",
    "probe_bridge_replay": "bridge with xeno-canto replay",
    "probe_perch": "Perch 2.0",
    "probe_birdnet": "BirdNET",
    "probe_biolingual": "BioLingual",
}
DETECTORS = ("whisperseg", "voxaboxen", "tweetynet", "das")
END_TO_END = ("cascade", "dualwhisper", "voxaboxen", "birdcode")
SIZE_HEADER = [str(size) for size in config.SIZES]


# ---------------------------------------------------------------- formatting
def format_cell(value):
    """Return a table cell as text: a float with four decimals, None as '--'."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "--"
    return str(value)


def table(title, header, rows):
    """Print a table; the first column is left-aligned, the other columns right-aligned."""
    lines = [header]
    for row in rows:
        lines.append([format_cell(value) for value in row])
    widths = []
    for column in range(len(header)):
        widths.append(max(len(line[column]) for line in lines))
    print(f"\n{title}")
    for line in lines:
        cells = [line[0].ljust(widths[0])]
        for cell, width in zip(line[1:], widths[1:]):
            cells.append(cell.rjust(width))
        print("  " + "  ".join(cells))


def by_size(values, key=None):
    """Return one entry of `values` per training size: the value itself, or its field `key`."""
    if key is None:
        return [values[size] for size in config.SIZES]
    return [values[size][key] for size in config.SIZES]


def end_to_end_by_size(scores, basis, key):
    """Return one end-to-end score per training size, on the given tolerance basis."""
    return [scores[size][basis][key] for size in config.SIZES]


def reported_basis(system):
    """Return the tolerance basis on which the end-to-end score of a system is reported."""
    if system == "birdcode":
        return "frame_tolerance"
    return "corpus_tolerance"


def interval(test):
    """Return a bootstrap result as 'difference [lower, upper]'."""
    lower, upper = test["interval"]
    return "{:+.4f} [{:+.4f}, {:+.4f}]".format(test["difference"], lower, upper)


def signed(value):
    """Return a number with its sign and four decimals."""
    return "{:+.4f}".format(value)


def rounded_by_name(p_values):
    """Return {printed system name: P value rounded to four decimals}."""
    return {NAMES[system]: round(p, 4) for system, p in p_values.items()}


def spread(distribution):
    """Return a distribution as 'min / median [q1-q3] / max, at cap count'."""
    return "{min} / {median:g} [{q1:g}-{q3:g}] / {max}, at cap {at_cap}".format(**distribution)


# ---------------------------------------------------------------- detection
def print_detection(det, e2e):
    """Print the main detection table and the tests that go with it."""
    rows = []
    for system in ("whisperseg", "whisperseg_no_pretraining"):
        rows.append([NAMES[system]] + by_size(det[system], "f1"))
    for system in ("dualwhisper", "cascade"):
        scores = end_to_end_by_size(e2e[system], "corpus_tolerance", "detection_f1")
        rows.append([NAMES[system]] + scores)
    for system in ("voxaboxen", "tweetynet", "das"):
        rows.append([NAMES[system]] + by_size(det[system], "f1"))
    for system in ("tweetynet", "das"):
        name = NAMES[system] + ", adjacent intervals merged"
        rows.append([name] + by_size(det[system], "f1_merged"))
    birdcode = end_to_end_by_size(e2e["birdcode"], "frame_tolerance", "detection_f1")
    rows.append(["BirdCODE (131.6 ms, nine corpora)"] + birdcode)
    rows.append([NAMES["energy"], det["energy"]["f1"], None, None])
    table(
        "Detection F1 (corpus-macro; DualWhisper, cascade and BirdCODE: "
        "on the Detection+Identification subset)",
        ["system"] + SIZE_HEADER,
        rows,
    )
    print(
        "  Wilcoxon signed-rank across corpora, WhisperSeg against each detector at 500:",
        rounded_by_name(det["wilcoxon_500"]),
    )
    changes = []
    for system in DETECTORS:
        for size in config.SIZES:
            scores = det[system][size]
            changes.append(abs(scores["f1_maximum"] - scores["f1"]))
    print(
        "  largest change of a detection F1 under maximum instead of greedy matching: "
        "{:.4f}".format(max(changes))
    )


def print_detection_details(det):
    """Print detection by corpus, precision and recall, and the secondary endpoints."""
    systems = DETECTORS + ("birdcode",)
    rows = []
    for corpus in config.CORPORA:
        row = [config.CORPUS_NAME[corpus]]
        for system in systems:
            row.append(det[system][500]["per_corpus"].get(corpus, {}).get("f1"))
        rows.append(row)
    means = [det[system][500]["f1"] for system in DETECTORS]
    rows.append(["Corpus mean"] + means + [det["birdcode"][500]["f1_frame_tolerance"]])
    table(
        "Detection F1 by corpus at 500 (BirdCODE: 131.6 ms)",
        ["corpus"] + [NAMES[system] for system in systems],
        rows,
    )

    header = ["system"]
    for size in SIZE_HEADER:
        header.append(f"P {size}")
        header.append(f"R {size}")
    rows = []
    for system in DETECTORS:
        row = [NAMES[system]]
        for size in config.SIZES:
            row.append(det[system][size]["precision"])
            row.append(det[system][size]["recall"])
        rows.append(row)
    table("Detection precision and recall against the common reference", header, rows)

    rows = []
    for system in DETECTORS:
        scores = det[system][500]
        rows.append([NAMES[system], scores["f1"], scores["iou_f1"], scores["frame_f1"]])
    table(
        "Secondary detection endpoints at 500",
        ["system", "two-boundary F1", "IoU-F1", "frame F1"],
        rows,
    )

    rows = []
    for system in ("tweetynet", "das"):
        rows.append([NAMES[system]] + by_size(det[system], "frame_f1"))
    table("Frame F1", ["system"] + SIZE_HEADER, rows)


def print_frame_tolerance(det, e2e):
    """Print the scores of all systems at the BirdCODE frame tolerance."""
    rows = []
    for system in DETECTORS:
        rows.append([NAMES[system]] + by_size(det[system], "f1_frame_tolerance") + [None] * 3)
    birdcode = det["birdcode"][500]["f1_frame_tolerance"]
    rows.append([NAMES["birdcode"], None, None, birdcode, None, None, None])
    for system in END_TO_END:
        detection_f1 = end_to_end_by_size(e2e[system], "frame_tolerance", "detection_f1")
        end_to_end_f1 = end_to_end_by_size(e2e[system], "frame_tolerance", "f1")
        rows.append([NAMES[system] + " (subset)"] + detection_f1 + end_to_end_f1)
    table(
        "All systems at 131.6 ms over nine corpora: detection F1 | end-to-end F1",
        ["system"] + SIZE_HEADER + SIZE_HEADER,
        rows,
    )


def print_merging(det):
    """Print the detection scores after merging adjacent vocalizations."""
    rows = []
    for system in DETECTORS:
        row = [NAMES[system]]
        for key in ("f1", "f1_merged", "f1_merged_others"):
            row += by_size(det[system], key)
        rows.append(row)
    table(
        "Detection F1 after merging adjacent vocalizations: "
        "reported | merged | merged, nine corpora",
        ["detector"] + SIZE_HEADER * 3,
        rows,
    )
    changes = []
    for system in DETECTORS:
        for size in config.SIZES:
            for corpus, scores in det[system][size]["merged_per_corpus"].items():
                if corpus not in analyses.CONTIGUOUS:
                    changes.append(scores["f1_merged"] - scores["f1"])
    print("  change on the nine corpora: {:+.4f} to {:+.4f}".format(min(changes), max(changes)))
    for corpus in analyses.CONTIGUOUS:
        scores = det["whisperseg"][500]["merged_per_corpus"][corpus]
        print(
            "  {}: {} references merged into {}; WhisperSeg F1 at 500 {:.4f} -> {:.4f}".format(
                config.CORPUS_NAME[corpus],
                scores["references"],
                scores["references_merged"],
                scores["f1"],
                scores["f1_merged"],
            )
        )


def print_restored(det, e2e):
    """Print the change in F1 when the long vocalizations are restored as references."""
    rows = []
    for system in ("whisperseg", "whisperseg_no_pretraining", "voxaboxen", "tweetynet", "das"):
        row = [NAMES[system], "detection"]
        for size in config.SIZES:
            scores = det[system][size]
            row.append(signed(scores["f1_restored"] - scores["f1"]))
        rows.append(row)
    for system in END_TO_END:
        row = [NAMES[system], "end-to-end"]
        for size in config.SIZES:
            scores = e2e[system][size]
            reported = scores[reported_basis(system)]["f1"]
            row.append(signed(scores["f1_restored"] - reported))
        rows.append(row)
    table(
        "Change in F1 with the long vocalizations restored as references",
        ["system", "endpoint"] + SIZE_HEADER,
        rows,
    )


# ---------------------------------------------------------------- identification
def print_identification(ident):
    """Print the identification scores of the systems and of the frozen probes."""
    system_rows = []
    probe_rows = []
    for system, scores in ident["macro_f1"].items():
        row = [NAMES[system]] + by_size(scores)
        if system.startswith("probe_"):
            probe_rows.append(row)
        else:
            system_rows.append(row)
    table("Identification macro-F1 (76 scored taxa)", ["system"] + SIZE_HEADER, system_rows)
    table(
        "Frozen encoders with an MLP probe, identification macro-F1",
        ["encoder"] + SIZE_HEADER,
        probe_rows,
    )

    rows = []
    for comparison, test in ident["probe_difference_500"].items():
        rows.append([comparison, interval(test), test["p"]])
    table(
        "Frozen probes at 500, paired bootstrap over test examples",
        ["comparison", "difference [95% interval]", "P"],
        rows,
    )

    rows = []
    for system, tests in ident["difference"].items():
        for size, test in sorted(tests.items(), reverse=True):
            rows.append([NAMES[system], size, interval(test), test["p"]])
    table(
        "WhisperID minus comparator, macro-F1 (paired bootstrap over test examples)",
        ["comparator", "size", "difference [95% interval]", "P"],
        rows,
    )


# ---------------------------------------------------------------- end-to-end
def print_end_to_end(e2e):
    """Print the end-to-end scores and their decomposition into detection and naming."""
    rows = []
    for system in END_TO_END:
        scores = end_to_end_by_size(e2e[system], reported_basis(system), "f1")
        rows.append([NAMES[system]] + scores)
    table("End-to-end F1 (BirdCODE: 131.6 ms, nine corpora)", ["system"] + SIZE_HEADER, rows)
    print(
        "  Wilcoxon signed-rank across corpora, cascade against each system at 500:",
        rounded_by_name(e2e["wilcoxon_500"]),
    )

    rows = []
    for system in END_TO_END:
        scores = e2e[system][500][reported_basis(system)]
        rows.append(
            [
                NAMES[system],
                scores["f1"],
                scores["detection_f1"],
                scores["conditional_accuracy"],
                scores["matched"],
            ]
        )
    table(
        "End-to-end error decomposition at 500",
        ["system", "end-to-end F1", "detection F1", "conditional taxon accuracy", "matched"],
        rows,
    )


# ---------------------------------------------------------------- overlap and tolerance
def print_overlap(overlap):
    """Print recall by overlap stratum and the differences to WhisperSeg."""
    strata = ("all", "isolated", "overlapping", "pairs")
    print(
        "\nOverlap detection at 500: {} reference vocalizations, {} overlapping pairs".format(
            overlap["vocalizations"], overlap["pairs"]
        )
    )
    rows = []
    for system, recall in overlap["recall"].items():
        rows.append([NAMES[system]] + [recall[stratum] for stratum in strata])
    table("Recall", ["system", "all", "isolated", "overlapping", "pairs resolved"], rows)

    p_values = set()
    rows = []
    for system, tests in overlap["difference"].items():
        for test in tests.values():
            p_values.add(test["p"])
        rows.append([NAMES[system]] + [interval(tests[stratum]) for stratum in strata])
    table(
        "WhisperSeg minus system [95% interval], all P = {}".format(sorted(p_values)),
        ["system"] + list(strata),
        rows,
    )


def print_short_recall(sweep):
    """Print the recall of short vocalizations at each common tolerance."""
    header = ["system", "vocalizations"]
    for tolerance in sweep["tolerances"]:
        header.append(str(round(1000 * tolerance)))
    rows = []
    for system, recall in sweep["recall"].items():
        rows.append([NAMES[system], sweep["vocalizations"][system]] + recall)
    table("Recall of vocalizations up to 200 ms by common tolerance (ms), at 500", header, rows)
    p_values = {NAMES[system]: p for system, p in sweep["p"].items()}
    print("  paired bootstrap over test chunks, WhisperSeg against each system:", p_values)


# ---------------------------------------------------------------- composition
def print_evaluation_sets(sets):
    """Print the sizes of the three evaluation sets."""
    identification = sets["identification"]
    detection = sets["detection"]
    end_to_end = sets["end_to_end"]
    rows = [
        [
            "Identification",
            identification["examples"],
            identification["examples"],
            identification["unscored"],
            identification["recordings"],
        ],
        [
            "Detection",
            detection["chunks"],
            detection["vocalizations"],
            None,
            detection["recordings"],
        ],
        [
            "Detection+Identification",
            end_to_end["chunks"],
            end_to_end["vocalizations"],
            end_to_end["unidentified"],
            None,
        ],
    ]
    table(
        "Evaluation sets",
        ["endpoint", "units", "reference vocalizations", "unscored / unidentified", "recordings"],
        rows,
    )


def print_training_sets(sets):
    """Print the composition of the training sets."""
    hours = []
    for value in by_size(sets, "detection_window_hours"):
        hours.append("{:.2f}".format(value))
    examples_per_taxon = []
    for distribution in by_size(sets, "identification_examples_per_taxon"):
        examples_per_taxon.append(spread(distribution))
    vocalizations_per_taxon = []
    for distribution in by_size(sets, "detection_vocalizations_per_taxon_and_corpus"):
        vocalizations_per_taxon.append(spread(distribution))
    minutes = []
    for value in by_size(sets, "detection_minutes_per_taxon_median"):
        minutes.append("{:.2f}".format(value))
    rows = [
        ["identification examples"] + by_size(sets, "identification_examples"),
        ["detection windows"] + by_size(sets, "detection_windows"),
        ["detection window audio (h)"] + hours,
        ["identification examples per scored taxon"] + examples_per_taxon,
        ["detection vocalizations per scored taxon and corpus"] + vocalizations_per_taxon,
        ["median annotated minutes per scored taxon, detection"] + minutes,
    ]
    table("Training sets", ["quantity"] + SIZE_HEADER, rows)


def print_tables(results):
    """Print every table of the paper from the results of score.py."""
    det = results["detection"]
    e2e = results["end_to_end"]
    print_detection(det, e2e)
    print_identification(results["identification"])
    print_end_to_end(e2e)
    print_detection_details(det)
    print_frame_tolerance(det, e2e)
    print_merging(det)
    print_restored(det, e2e)
    print_overlap(results["overlap"])
    print_short_recall(results["short_recall"])
    print_evaluation_sets(results["evaluation_sets"])
    print_training_sets(results["training_sets"])
