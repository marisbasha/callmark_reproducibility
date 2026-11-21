# CallMark: code to reproduce the experiments

Code for the experiments of the CallMark paper. Every folder has its own README with the
environment, the data layout, the commands in order and what each command writes.

| folder | content | paper |
|---|---|---|
| `benchmark/` | scoring of the cross-taxon benchmark of 11 corpora: detection, identification, Detection+Identification and overlap detection, with the statistical tests | benchmark figure and tables |
| `systems/` | data preparation, training and prediction of every system scored in the benchmark | benchmark figure and tables |
| `correction/` | boundary correction on synthetic stimuli annotated by eight annotators | boundary-correction figure and table |
| `multichannel/` | vocalization detection with body-mounted sensors (MC-WhisperSeg), and the microphone-only detector trained on group audio labelled by MC-WhisperSeg | Fig. 4, Extended Data figure, Supplementary Table |

## Benchmark systems

`systems/<name>/` writes the prediction files that `benchmark/` scores. Scripts run as modules
from the repository root, for example `python -m systems.tweetynet.train --size 50`.

| folder | system | tasks |
|---|---|---|
| `sources/` | download of the source corpora, BirdSet XCL and xeno-canto; build of the dataset from the source corpora | |
| `data/` | training windows, test chunks and label files from the dataset | |
| `whisperseg/` | WhisperSeg, with and without detection pretraining | detection |
| `whisperid/` | WhisperID: pretraining, fine-tuning, frozen-encoder embeddings | identification |
| `cascade/` | WhisperSeg, then WhisperID | Detection+Identification |
| `dualwhisper/` | DualWhisper | Detection+Identification, identification |
| `voxaboxen/` | Voxaboxen | detection, Detection+Identification, identification |
| `birdcode/` | BirdCODE, fine-tuned | detection, Detection+Identification, identification |
| `tweetynet/`, `das/` | TweetyNet, DAS | detection |
| `energy/` | energy-threshold detector | detection |
| `biolingual/`, `naturelm/` | BioLingual (fine-tuned and frozen), NatureLM-audio with adapters | identification |
| `frozen_encoders/` | Perch and BirdNET embeddings for the probes of `benchmark/probe.py` | identification |

Training on a GPU is not bit-reproducible; several systems do not seed the order of their
training batches. The READMEs state which results were reproduced exactly from the trained
models and which only within the variation between runs.

## Data

Every script reads one data root, given by the environment variable `CALLMARK_DATA` or by `--data`.

- `correction/` includes its data.
- `multichannel/`: the public group recordings and their annotations are on Zenodo
  (DOI 10.5281/zenodo.13144875). The recordings of pairs used for training and the unannotated
  group recordings are available from the corresponding author.
- `benchmark/` and `systems/`: the dataset and the prediction files are available from the
  corresponding author during peer review and will be archived with the published article.
  `systems/sources/` lists the public origin of every source corpus.

## Trained models

The trained models of the paper are on Hugging Face, in the collection "CallMark reproducibility"
of https://huggingface.co/songbirdini. `download_models.py` places them in the data root where the
prediction scripts read them:

    python download_models.py all --data /path/to/data
    python download_models.py whisperseg whisperid --data /path/to/data

| Name | Repository | Size |
|---|---|---|
| `whisperseg` | `songbirdini/callmark-whisperseg` | 51 GB |
| `whisperid` | `songbirdini/callmark-whisperid` | 46 GB |
| `dualwhisper` | `songbirdini/callmark-dualwhisper` | 22 GB |
| `voxaboxen` | `songbirdini/callmark-voxaboxen` | 6 GB |
| `birdcode` | `songbirdini/callmark-birdcode` | 2 GB |
| `naturelm` | `songbirdini/callmark-naturelm` | 47 GB |
| `biolingual` | `songbirdini/callmark-biolingual` | 2 GB |
| `tweetynet` | `songbirdini/callmark-tweetynet` | 4 MB |
| `das` | `songbirdini/callmark-das` | 6 MB |
| `mc-whisperseg` | `songbirdini/callmark-mc-whisperseg` | 18 GB |

`mc-whisperseg` holds MC-WhisperSeg, the two single-sensor models and the microphone-only detector
trained with added group audio (`multichannel/`). The cascade uses `whisperseg` and `whisperid`.
The energy detector and the frozen-encoder probes need no trained model from here.

## Third-party code

The repository contains no copies of unmodified third-party code. Each system README names the
upstream repository and commit. Modified files are included where the upstream licence allows:

- `multichannel/mcwhisperseg/`: modified from https://github.com/nianlonggu/MCWhisperSeg
  (Apache-2.0, `multichannel/mcwhisperseg/LICENSE`).
- `systems/voxaboxen/voxaboxen_changes/`: five files modified from
  https://github.com/earthspecies/voxaboxen, distributed under its licence, AGPL-3.0
  (`systems/voxaboxen/voxaboxen_changes/LICENSE`), not under the licence below.
- `systems/whisperseg/whisperseg.patch`: changes to https://github.com/nianlonggu/WhisperSeg,
  applied to a clone.

## Licence

Copyright 2026 The CallMark authors. Released under the Apache License 2.0 (`LICENSE`), except
`systems/voxaboxen/voxaboxen_changes/` (AGPL-3.0).
