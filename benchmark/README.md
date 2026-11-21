# Cross-taxon benchmark: scoring

This directory scores the benchmark of 11 corpora and 76 scored taxa: detection, identification,
Detection+Identification (end-to-end) and overlap detection, with the statistical tests and the supplementary
analyses. It starts from the published dataset and from one prediction file per system and training size.

    python -m benchmark.probe --data DIR      # frozen-encoder probes: embeddings -> identification predictions
    python -m benchmark.score --data DIR      # all scores -> DIR/results/benchmark.json, tables on standard output

Run from the parent directory of `benchmark/`. `--data` can be replaced by the environment variable
`CALLMARK_DATA`. Requirements: numpy, scipy, scikit-learn, pyarrow, soundfile.

| module | content |
|---|---|
| `config.py` | corpora, tolerances, training sizes, file locations |
| `data.py` | dataset rows, reference annotations, prediction files |
| `metrics.py` | matching; detection, end-to-end and identification scores |
| `stats.py` | paired bootstraps, Wilcoxon signed-rank test across corpora |
| `analyses.py` | merged adjacent vocalizations, restored long vocalizations, overlap, tolerance sweep |
| `composition.py` | evaluation sets and training-set sizes |
| `probe.py` | MLP probe on frozen embeddings |
| `score.py`, `report.py` | everything above for every system; result file and tables |

## Data root

    dataset/classification/{train_50,train_100,train_500,test}-*.parquet    identification examples
    dataset/segmentation/{train_50,train_100,train_500,test}-*.parquet      detection chunks
    prepared/test/<chunk_id>.wav, .json            detection test set (onset, offset in seconds from chunk start)
    prepared/train_<size>/<window>.wav, .json      detection training windows; "cluster" holds the taxon of each
                                                   vocalization ("unknown": unidentified)
    labels/test_taxa.json                          {chunk_id: [taxon or null, ...]}, one entry per annotation of the
                                                   test chunk in dataset order
    labels/end_to_end_chunks.json                  [chunk_id, ...], the Detection+Identification subset
    labels/excluded_long_test.json                 {chunk_id: [[onset, offset, taxon], ...]}, vocalizations removed
                                                   by the duration filter that overlap a test chunk
    embeddings/<encoder>.npy                       one row per identification example: train_500, then test
    predictions/<system>/detection_<size>.json.gz
    predictions/<system>/end_to_end_<size>.json.gz
    predictions/<system>/identification_<size>.json
    results/benchmark.json

`<size>` is 50, 100 or 500. Rows of the corpus `rodents2019` (the recordings of `rodent_usv` a second time) are
ignored wherever the dataset is read.

## Prediction files

`detection_<size>.json.gz`: a JSON list with one entry for every chunk of `dataset/segmentation/test`,

    {"chunk_id": "...", "pred": [[onset, offset], ...]}

with times in seconds from the start of the chunk, in the order in which the system emitted them (matching is
greedy in this order).

`end_to_end_<size>.json.gz`: the same for the chunks of `labels/end_to_end_chunks.json`, with
`"pred": [[onset, offset, taxon], ...]`.

`identification_<size>.json`: a JSON object `{example_id: taxon}` with one entry for every example of
`dataset/classification/test`; `example_id` is `<chunk_id>__e<event_index, four digits>`.

| system | detection | end_to_end | identification |
|---|---|---|---|
| `whisperseg` | 50, 100, 500 | | |
| `whisperseg_no_pretraining` | 50, 100, 500 | | |
| `whisperid` | | | 50, 100, 500 |
| `cascade` (WhisperSeg, then WhisperID) | | 50, 100, 500 | |
| `dualwhisper` | | 50, 100, 500 | 50, 100, 500 |
| `voxaboxen` | 50, 100, 500 (detection model) | 50, 100, 500 (taxon model) | 50, 100, 500 |
| `tweetynet`, `das` | 50, 100, 500 | | |
| `birdcode` | 500 | 50, 100, 500 | 50, 100, 500 |
| `biolingual`, `naturelm` | | | 50, 100, 500 |
| `energy` | `detection.json.gz` (no training) | | |
| `probe_<encoder>` | | | 50, 100, 500, written by `probe.py` |

Encoders: `whisperseg` (released checkpoint), `whisper_large`, `whisperid_bridge`, `whisperid_bridge_xcl1`,
`whisperid_pretrained`, `xeno_canto_only`, `bridge_replay`, `perch`, `birdnet`, `biolingual`.

## Scoring rules

Detection: a prediction matches a reference vocalization when both boundaries lie within the corpus tolerance
(`config.TOLERANCE`); F1 per corpus, then the unweighted mean over the 11 corpora. WhisperSeg is scored against the
annotations of the prepared test set, the other detectors against the annotations of the dataset (the same
annotations, in the prepared test set with offsets clipped to the end of the audio); precision and recall, the merge
analysis, overlap detection and the tolerance sweep use the prepared test set for every system. BirdCODE is scored
with both boundaries within one output frame (131.6 ms) over the nine corpora within its 32-kHz input.

Identification: macro-F1 over the 76 scored taxa on the 9,552 test examples of the 11 corpora; the frozen-encoder
probes are scored on all test examples of the dataset.

Detection+Identification: as detection, with the taxon required to agree, on the chunks of
`labels/end_to_end_chunks.json`.
