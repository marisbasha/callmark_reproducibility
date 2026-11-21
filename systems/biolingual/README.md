# BioLingual

BioLingual (`davidrrobinson/BioLingual`, a CLAP model trained on animal sounds) in two uses:
fully fine-tuned for identification with a linear layer over the 160 taxa of the dataset, and
unchanged as a frozen encoder for the probes of `benchmark/probe.py`.

## Environment

Python 3.11 with

| package | version |
|---|---|
| torch | 2.13.0 |
| transformers | 4.44.2 |
| numpy | 2.4.6 |
| pyarrow | 25.0.0 |
| soundfile | 0.14.0 |

Run from the repository root; `--data` can be replaced by the environment variable
`CALLMARK_DATA`. `clips.py` of `systems/frozen_encoders` reads the examples.

## Input

    dataset/classification/{train_50,train_100,train_500,test}-*.parquet

## Commands

    python -m systems.biolingual.train --data DIR --size 500      # also 50 and 100
    python -m systems.biolingual.predict --data DIR --size 500
    python -m systems.biolingual.embed --data DIR

| command | writes | hardware and run time |
|---|---|---|
| `train` | `models/biolingual/train_<size>.pt` | RTX A6000: 17, 26 and 55 min for sizes 50, 100 and 500 |
| `predict` | `predictions/biolingual/identification_<size>.json` | 8 processor threads: 1 h per size |
| `embed` | `embeddings/biolingual.npy` (27,628 x 512; `train_500`, then `test`) | RTX 3090: 14 min |

## Fine-tuning

| setting | value |
|---|---|
| input | the vocalization (onset to offset), resampled to 48 kHz by linear interpolation |
| trained weights | the whole model and the linear layer (153.6 million parameters) |
| validation set | a tenth of the recordings of every taxon with at least two recordings, seed 42 |
| augmentation | gain within 6 dB, Gaussian noise at 20 to 40 dB signal-to-noise ratio, circular shift of up to 10 %, sign flip; drawn once per training example and repeated in every epoch |
| batch | 16 |
| optimizer | AdamW, weight decay 0.01, gradient norm clipped to 1 |
| learning rate | one cycle over 50 epochs, 10 % rising; maximum 2e-5 for the model, 1e-3 for the linear layer |
| precision | bfloat16 autocast |
| stopping | after 8 epochs without a higher validation macro-F1, at most 50 epochs |
| kept model | the epoch with the highest validation macro-F1 over the taxa marked `scorable` |
| seed | 42 |

| training size | training examples | validation examples | best epoch | validation macro-F1 |
|---|---|---|---|---|
| 50 | 3,946 | 544 | 26 | 0.6018 |
| 100 | 6,382 | 859 | 24 | 0.5641 |
| 500 | 15,659 | 2,127 | 19 | 0.6592 |

Prediction runs on the processor with bfloat16 autocast in batches of 32.
