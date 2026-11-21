# Frozen encoders: Perch and BirdNET embeddings

Embeddings of the published Perch and BirdNET models for the frozen-encoder probes
(`benchmark/probe.py`). Both models are used unchanged.

## Environment

Python 3.11 with

| package | version |
|---|---|
| numpy | 2.4.6 |
| tensorflow | 2.20.0 |
| perch-hoplite | 1.0.2 |
| birdnetlib | 0.18.1 |
| librosa | 0.11.0 |
| pyarrow | 25.0.0 |
| soundfile | 0.14.0 |

`perch-hoplite` downloads Perch v2 (`google/bird-vocalization-classifier`, variation
`perch_v2`, version 2) from Kaggle on first use. `birdnetlib` contains the BirdNET model
(`BirdNET_GLOBAL_6K_V2.4_Model_FP32`). Run from the repository root; `--data` can be replaced by
the environment variable `CALLMARK_DATA`.

## Input

    dataset/classification/{train_500,test}-*.parquet

## Commands

    python -m systems.frozen_encoders.embed_perch --data DIR
    python -m systems.frozen_encoders.embed_birdnet --data DIR

| command | writes | rows x dimensions | run time |
|---|---|---|---|
| `embed_perch` | `embeddings/perch.npy` | 27,628 x 1,536 | 5 min (RTX 3090) |
| `embed_birdnet` | `embeddings/birdnet.npy` | 27,628 x 1,024 | 17 min (one processor thread) |

Row order: the 17,786 examples of `train_500`, then the 9,842 examples of `test`, in dataset
order. The embeddings of the smaller training sizes are the rows with `fewshot_rank` below the
size.

## Audio given to the encoders

`clips.py` cuts the vocalization (onset to offset) out of the 5-s window of the example and
resamples it by linear interpolation.

| encoder | sampling rate | input |
|---|---|---|
| Perch | 32 kHz | the vocalization at its own length; the embeddings of all frames are averaged |
| BirdNET | 48 kHz | one 3-s window: a shorter vocalization in the centre of zeros, of a longer one the central 3 s |

The embeddings of BioLingual and of the Whisper-based encoders are written by
`systems/biolingual/embed.py` and `systems/whisperid/embed.py`.
