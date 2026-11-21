# Voxaboxen

Voxaboxen (https://github.com/earthspecies/voxaboxen) predicts onset, duration and class of
every vocalization with a BirdAVES encoder, once forward and once backward in time, and combines
the two sets of boxes. It is trained here three times per training size:

| model | labels | prediction file |
|---|---|---|
| detection | every vocalization is `Vocal` | `predictions/voxaboxen/detection_<size>.json.gz` |
| taxon | taxon of every vocalization | `predictions/voxaboxen/end_to_end_<size>.json.gz` |
| identification | one vocalization per clip, with its taxon | `predictions/voxaboxen/identification_<size>.json` |

## Environment

- Voxaboxen at commit `20cf96636620aba466d393fd84ffb272667d3611`, with the five files of
  `voxaboxen_changes/` copied over the checkout:

      git clone https://github.com/earthspecies/voxaboxen
      cd voxaboxen && git checkout 20cf96636620aba466d393fd84ffb272667d3611
      cp -r <this folder>/voxaboxen_changes/voxaboxen .
      uv sync

- Python 3.11 with the packages of Voxaboxen's `uv.lock` (torch 2.7.0, torchaudio 2.7.0,
  numpy 2.2.6, pandas 2.2.3, librosa 0.11.0, soundfile 0.13.1) and pyarrow.
- The BirdAVES weights (`birdaves-biox-base.torchaudio.pt`, 360 MB) are downloaded by
  Voxaboxen at the first training.
- One GPU with 12 GB.

Run the scripts with the Python of this environment from the repository root. `--data` can be
replaced by the environment variable `CALLMARK_DATA`; `--voxaboxen` is the checkout.

## Data

Read from the data root (`benchmark/README.md`, `systems/data/README.md`):

    prepared/train_<size>_joined/    joined training windows (python -m systems.data.join_windows)
    prepared/test/                   detection test set
    dataset/classification/          identification examples
    dataset/segmentation/test-*      annotations of the test chunks
    labels/test_taxa.json, labels/end_to_end_chunks.json

Written to the data root:

    voxaboxen/<model>_<size>/        Voxaboxen input: audio, selection tables, file lists
    models/voxaboxen/<model>_<size>/ Voxaboxen project with the training run in run/
    predictions/voxaboxen/           prediction files

## Commands

For each `<size>` of 50, 100 and 500:

    python -m systems.voxaboxen.prepare_detection --data DIR --size <size> --model detection
    python -m systems.voxaboxen.train --data DIR --voxaboxen CHECKOUT --model detection --size <size>
    python -m systems.voxaboxen.predict_detection --data DIR --size <size>

    python -m systems.voxaboxen.prepare_detection --data DIR --size <size> --model taxon
    python -m systems.voxaboxen.train --data DIR --voxaboxen CHECKOUT --model taxon --size <size>
    python -m systems.voxaboxen.predict_end_to_end --data DIR --voxaboxen CHECKOUT --size <size>

    python -m systems.voxaboxen.prepare_identification --data DIR --size <size>
    python -m systems.voxaboxen.train --data DIR --voxaboxen CHECKOUT --model identification --size <size>
    python -m systems.voxaboxen.predict_identification --data DIR --size <size>

`predict_end_to_end` needs the detection model of the same size to be trained first.

| script | writes |
|---|---|
| `prepare_detection` | `voxaboxen/detection_<size>/` or `voxaboxen/taxon_<size>/`: `selection_tables/`, `audio/` (slowed ultrasonic files), `train_info.csv`, `val_info.csv`, `test_info.csv`, `test_slow_down.json` |
| `prepare_identification` | `voxaboxen/identification_<size>/`: `train_audio/`, `train_tables/`, `test_audio/`, `test_tables/`, the three file lists, `test_examples.json` |
| `train` | `models/voxaboxen/<model>_<size>/`: `project_config.yaml`, `train.log`, `run/final-model.pt`, `run/params.yaml`, `run/outputs/` (predictions for the validation and test files) |
| `predict_detection` | `predictions/voxaboxen/detection_<size>.json.gz` |
| `predict_end_to_end` | `models/voxaboxen/taxon_<size>/run/inference/`, `predictions/voxaboxen/end_to_end_<size>.json.gz` |
| `predict_identification` | `predictions/voxaboxen/identification_<size>.json` |

Run time of `train` (50 epochs and the prediction of the validation and test files) on one
TITAN X (Pascal), RTX 3090 or RTX A6000:

| model | size 50 | size 100 | size 500 |
|---|---|---|---|
| detection | 2.1 h | 2.8 h | 7.1 h |
| taxon | 3.3 h | 3.5 h | 11.6 h |
| identification | 3.8 h | 4.6 h | 7.2 h |

The preparation takes a few minutes per model and size.

## Input of the detection and taxon models

- **Slow-down.** Voxaboxen resamples to 16 kHz. Files that declare a minimum frequency
  (`rodent_usv`, `singing_mice`) are written with the sampling rate in the header divided by
  the smallest power of two that brings it to 16 kHz or below (16 at 250 kHz), and their
  annotation times are multiplied by the same factor. Predicted times of test chunks are
  divided by the factor (60 test chunks).
- **Labels.** Detection model: `Vocal`. Taxon model: over the joined training files in name
  order, the first `<size>` vocalizations of a taxon keep their taxon, counted over all corpora
  together; further ones (102, 188 and 500 at sizes 50, 100 and 500) and unidentified ones are
  labelled `Unknown`, which Voxaboxen trains for detection only. Voxaboxen takes the classes
  from the training, validation and test tables.
- **Validation files.** A tenth of the joined files of every corpus (at least one, never all;
  none of a corpus with a single file), drawn with `numpy.random.RandomState` seeded with the
  4-byte BLAKE2b digest of `42:<corpus>` modulo 2^31.
- **Files without a training clip.** Voxaboxen cuts `floor((duration - 10 s) / 5 s)` training
  clips of 10 s from a file, so a file shorter than 15 s gives none and is left out of both
  lists.
- **Clip budget of slowed corpora.** In the training list, the files of a corpus are taken in
  order until their clips reach the number of clips the corpus would give at original speed.
  Whole files are taken, and the first file is taken whenever the budget is at least one clip.
  The validation list has no budget.

Resulting file lists (files with at least one clip; `rodent_usv` 177, 169 and 169 clips from
one file each):

| size | training files | validation files | corpora without a training file |
|---|---|---|---|
| 50 | 278 | 33 | `singing_mice`, `spiders`, `wytham` |
| 100 | 445 | 54 | `singing_mice` |
| 500 | 1,072 | 122 | none |

The test list holds all 11,085 test chunks unchanged, one file per chunk.

## Input of the identification model

- Every example of `dataset/classification/train_<size>` and `test` is cut to its vocalization
  and centred in 3 s of silence (a vocalization of 3 s or more: 0.5 s of silence on each side).
  The clip carries one annotation, the vocalization with its taxon.
- Recordings at 200 kHz and above are slowed as above; the annotation is given on the slowed
  time axis.
- Validation clips: for every taxon with at least two recordings, a tenth of its recordings (at
  least one, never all) with all their examples, drawn with `random.Random(42)`.

## Training

`train` runs `main.py project-setup` and `main.py train-model` of the checkout with

| setting | value |
|---|---|
| encoder | BirdAVES (`--encoder-type=aves`), frozen during the first 3 epochs |
| directions | forward and backward (`--bidirectional`) |
| epochs | 50, no early stopping, weights of the last epoch |
| batch | 16 clips, gradients accumulated over 2 batches (`--batch-size=16 --grad-accum=2`) |
| learning rate | 5e-5, cosine schedule (Voxaboxen default) |
| clips | 10 s every 5 s; identification model: 1 s every 0.5 s |
| seed | 0 (Voxaboxen default) |
| data-loading processes | 4 for training; the per-file loaders of the evaluation run in the main process (`VOXABOXEN_EVAL_WORKERS=0`) |

All other settings are the defaults of Voxaboxen and are recorded in `run/params.yaml`.

## Thresholds and predictions

- Detection peaks at probability 0.5 or above become boxes; the class of a box is its most
  probable class (classification threshold 0).
- Forward and backward boxes are matched at an intersection over union of 0.5. After training,
  Voxaboxen chooses the threshold on the combined detection probability among 19 values from 0
  to 0.95 by the macro-F1 of the validation files at an intersection over union of 0.8, and
  writes the test predictions at that threshold to `run/outputs/`.

| model | chosen threshold at sizes 50, 100, 500 | used for the prediction file |
|---|---|---|
| detection | 0.792, 0.739, 0.792 | yes |
| taxon | 0.950, 0.950, 0.950 | no: the threshold of the detection model of the same size |
| identification | 0.000, 0.000, 0.686 | yes |

- `predict_end_to_end` runs `main.py inference` with the taxon model on the chunks of
  `labels/end_to_end_chunks.json` and combines the directions at the threshold of the detection
  model, as printed with three decimals in its `train.log`.
- `predict_identification` gives an example the label of the predicted box with the largest
  intersection over union with the vocalization in its clip, and `__none__` if no box overlaps
  it (87, 15 and 3 of 9,842 examples).
- Predictions are written in the order of Voxaboxen's selection tables.

## Voxaboxen code

`voxaboxen_changes/voxaboxen/` contains `data/data.py`, `evaluation/evaluation.py`,
`inference/inference.py`, `training/params.py` and `training/train.py` of
https://github.com/earthspecies/voxaboxen (commit 20cf966), distributed under the GNU Affero
General Public License 3.0 (`voxaboxen_changes/LICENSE`). Changes:

- `training/params.py`, `training/train.py`: option `--grad-accum K`. The loss of a batch is
  divided by K, gradients are reset before every K-th batch of an epoch and the optimizer steps
  after every K-th batch.
- `data/data.py`: the environment variable `VOXABOXEN_EVAL_WORKERS` sets the number of loading
  processes of the per-file loaders used for evaluation and inference.
- `inference/inference.py`: the combination of forward and backward predictions is called with
  the argument names of `combine_fwd_bck_preds` and reads the tables from the inference folder.
- `evaluation/evaluation.py`: `combine_fwd_bck_preds` also reads forward and backward tables
  named without the detection threshold, as inference writes them.

All other files of Voxaboxen are used unchanged.
