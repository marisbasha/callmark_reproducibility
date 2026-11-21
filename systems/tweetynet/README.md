# TweetyNet

TweetyNet (Cohen et al. 2022): a convolutional and recurrent network that labels every
spectrogram frame. One model is trained for all corpora on audio resampled to 32 kHz and
labels every 5-ms frame as noise or vocal.

Loading of the training files, the validation split, resampling and the conversion of frame
probabilities into vocalizations are in `../das/data.py`.

## Environment

Python 3.11, vak 1.0.4.post1 (`pip install vak==1.0.4.post1`, https://github.com/vocalpy/vak,
unmodified; only the network `vak.nets.TweetyNet` is used), PyTorch 2.13.0, numpy 2.4.6,
soundfile 0.14.0, pyarrow 25.0.0.

## Data

    dataset/segmentation/test-*.parquet     test chunks
    prepared/train_<size>_joined/           training files, written by
                                            python -m systems.data.join_windows (../data/README.md)

Only the audio and the `onset` and `offset` of the training files are used.

## Commands

Run from the repository root; `--data` can be replaced by the environment variable
`CALLMARK_DATA`. `<size>` is 50, 100 or 500.

    python -m systems.tweetynet.train --data DIR --size <size>
    python -m systems.tweetynet.predict --data DIR --size <size>

| command | writes |
|---|---|
| `train` | `models/tweetynet/<size>/best.pt` |
| `predict` | `predictions/tweetynet/detection_<size>.json.gz` |

## Method

| step | setting |
|---|---|
| audio | channels averaged, resampled to 32 kHz by linear interpolation |
| spectrogram | Hann window of 1,024 samples, hop of 160 samples (5 ms); power summed in 128 triangular bands with centres spaced evenly from 0 to 16 kHz; natural logarithm; the median over time of the file or chunk is subtracted from every band |
| target | per frame: vocal if an annotation touches the frame, noise otherwise |
| validation split | per corpus, a tenth of the training files (at least one, chosen with a fixed seed) is held out; a corpus with a single file is used for training only |
| training input | non-overlapping windows of 176 frames |
| network | `vak.nets.TweetyNet` with default settings, 128 frequency bins, 2 classes |
| training | Adam, learning rate 0.001, batch size 32, seed 42; cross-entropy with the vocal class weighted by the number of noise frames per vocal frame (at most 50); at most 60 epochs |
| model selection | epoch with the highest F1 of the vocal class over all validation frames; training stops after 8 epochs without improvement |
| detection | consecutive windows of 176 frames per test chunk (a last window shorter than 8 frames gets probability 0); a vocalization is every run of frames with vocal probability of at least 0.5, without filling of gaps and without a minimum duration |

| size | training windows | validation windows | corpora with validation files | selected epoch (first: 0) | epochs run |
|---|---|---|---|---|---|
| 50 | 12,402 | 1,543 | 6 of 13 | 6 | 15 |
| 100 | 20,082 | 2,517 | 9 of 13 | 10 | 19 |
| 500 | 55,835 | 6,314 | 13 of 13 | 14 | 23 |

## Run time

Training on one GeForce RTX 3090 with detection on the test set: 12 min (size 50), 21 min
(100), 24 min (500). The predictions were computed on the CPU (8 threads): 16 to 21 min for
the 11,085 test chunks.
