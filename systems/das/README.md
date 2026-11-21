# DAS

Deep Audio Segmenter (Steinfath et al. 2021): a temporal convolutional network on the raw
waveform. One model is trained for all corpora on audio resampled to 32 kHz and labels every
sample as noise or vocal.

`data.py` is shared with `../tweetynet/`.

## Environment

Python 3.11, das 0.32.9 (`pip install das==0.32.9`, https://github.com/janclemenslab/das,
unmodified), TensorFlow 2.20.0, Keras 3.15.1, numpy 2.4.6, soundfile 0.14.0, pyarrow 25.0.0.
DAS is written for Keras 2; `network.py` holds the two adjustments needed to run it with
Keras 3.

## Data

    dataset/segmentation/test-*.parquet     test chunks
    prepared/train_<size>_joined/           training files, written by
                                            python -m systems.data.join_windows (../data/README.md)

Only the audio and the `onset` and `offset` of the training files are used.

## Commands

Run from the repository root; `--data` can be replaced by the environment variable
`CALLMARK_DATA`. `<size>` is 50, 100 or 500.

    python -m systems.das.train --data DIR --size <size>
    python -m systems.das.predict --data DIR --size <size>

| command | writes |
|---|---|
| `train` | `models/das/<size>/das_model.h5`, `das_params.yaml` |
| `predict` | `predictions/das/detection_<size>.json.gz` |

## Method

| step | setting |
|---|---|
| audio | channels averaged, resampled to 32 kHz by linear interpolation |
| target | per sample: vocal inside an annotation, noise elsewhere |
| validation split | per corpus, a tenth of the training files (at least one, chosen with a fixed seed) is held out; a corpus with a single file is used for training only |
| network | DAS `tcn`: 4 blocks of separable convolutions, 32 filters, kernel size 16, input of 4,096 samples; 144,770 parameters |
| training | 40 epochs of at most 1,000 batches of 64 inputs; Adam, learning rate 0.0005, divided by 10 after 5 epochs without improvement of the validation loss; DAS defaults otherwise; no random seed is set |
| model selection | epoch with the lowest validation loss |
| detection | each test chunk on its own; a vocalization is every run of samples with vocal probability of at least 0.5, without filling of gaps and without a minimum duration (the DAS default) |

| size | training files | validation files | training audio (h) | validation audio (h) |
|---|---|---|---|---|
| 50 | 282 | 33 | 3.07 | 0.38 |
| 100 | 449 | 54 | 4.97 | 0.62 |
| 500 | 1,081 | 122 | 13.77 | 1.56 |

## Run time

One GeForce RTX 3090: training 3.5 h (size 50), 3.9 h (100), 5.0 h (500). Detection on the
11,085 test chunks takes 40 min. Training holds the training arrays in memory and writes them
to `models/das/<size>/training.npy/` until it ends (7 GB at size 500).
