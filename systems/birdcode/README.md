# BirdCODE, fine-tuned

BirdCODE is a pretrained frame-level sound event detector for bird vocalizations (Earth Species
Project). The released model is fine-tuned as a whole on the benchmark training sets: once on the
detection training windows, for detection and Detection+Identification, and once on the
identification examples, for identification.

| | |
|---|---|
| model code | https://github.com/earthspecies/sound-event-detection, commit `617c6408fdd1b56c3873ef2d17006ffbaee61783` (MIT), used unmodified |
| pretrained weights | Hugging Face Hub, `EarthSpeciesProject/sed-birdcode`, revision `fd16e12a98360db51b4b6e9c23e29309a804661d`; downloaded by the scripts on first use (about 1.1 GB) |
| input | mono audio at 32 kHz in windows of 5 s; every recording is resampled to 32 kHz |
| output | one probability per class every 131.6 ms (7.6 frames per second); 9,422 classes, 20 prototypes per class |

## Files

| file | purpose |
|---|---|
| `birdcode.py` | loading of the released and of the fine-tuned models, audio loading, clip with a centred vocalization |
| `prepare_identification_clips.py` | identification training examples as 5-s clips |
| `train.py` | fine-tuning |
| `predict.py` | prediction files for detection, Detection+Identification and identification |

## Environment

Python 3.12 or later with the environment of the model repository: clone it at the commit above
and run `uv sync --group gpu`. Versions used: PyTorch 2.11.0 (CUDA 12.6), torchaudio 2.11.0,
librosa 0.11.0, soundfile 0.14.0, numpy 2.4.6, pyarrow 25.0.0, pandas 3.0.3. Training uses
bfloat16 autocast and needs a GPU with CUDA compute capability 8.0 or higher.

Run the commands from the root of this repository with that environment active. `--data` can be
replaced by the environment variable `CALLMARK_DATA`.

## Data

Read from the data root of `benchmark/README.md`:

    dataset/classification/{train_<size>,test}-*.parquet    identification examples
    prepared/train_<size>/<window>.wav, .json               detection training windows with taxa
    prepared/test/<chunk_id>.wav, .json                     detection test set
    labels/end_to_end_chunks.json                           Detection+Identification subset

Written:

    birdcode/identification_clips_<size>/clips/<corpus>__<row index>.wav, .json
    birdcode/identification_clips_<size>/validation.txt
    models/birdcode/{detector,identifier}_<size>/best_model.pt, labels.txt, train_log.json
    predictions/birdcode/{detection,end_to_end}_<size>.json.gz, identification_<size>.json

## Commands

    python -m systems.birdcode.train --data DIR --model detector --size 500
    python -m systems.birdcode.predict --data DIR --task detection --size 500
    python -m systems.birdcode.predict --data DIR --task end_to_end --size 500

    python -m systems.birdcode.prepare_identification_clips --data DIR --size 500
    python -m systems.birdcode.train --data DIR --model identifier --size 500
    python -m systems.birdcode.predict --data DIR --task identification --size 500

Repeat with `--size 50` and `--size 100`. The benchmark scores detection at size 500 only.

## Training sets

| | detector | identifier |
|---|---|---|
| material | the files of `prepared/train_<size>`, one file per training window | every example of `dataset/classification/train_<size>`, the vocalization cut out and centred in 5 s of silence (longer vocalizations are kept whole) |
| left out | right whale and spider recordings; corpora sampled above 64 kHz (`rodent_usv`, `singing_mice`) | nothing |
| validation | a random tenth of the files (seed 42) | per taxon, a tenth of its recordings with all their examples (seed 42); taxa with one recording are not held out |
| files, training / validation, at 50, 100, 500 | 1,151 / 127, 1,647 / 183, 2,740 / 304 | 3,946 / 544, 6,382 / 859, 15,659 / 2,127 |
| output classes | 138 taxa, 4 of them unknown to the released model | 144 taxa, 10 of them unknown to the released model |

## Fine-tuning

- Output classes: the taxa of the identified vocalizations of the training set. A taxon among the
  classes of the released model keeps its prototypes, class weights and bias. Any other taxon
  starts with the prototypes of a randomly drawn class plus Gaussian noise (standard deviation
  0.01), the mean class weights and the mean bias.
- Targets on the output frames of the model (38 frames per 5-s window): a frame is positive for a
  taxon when a vocalization of that taxon covers at least half of the frame, or at least half of
  the vocalization lies in the frame. Frames touched by an unidentified vocalization carry no loss
  unless they are positive for a taxon.
- Loss: binary cross-entropy over frames and taxa.
- Every epoch draws one random 5-s window from every training file (shorter files are padded with
  zeros). Validation uses the first 5 s of every validation file.
- All parameters are trained: AdamW, learning rate 1e-5, weight decay 0.01, batch size 16,
  gradient norm clipped at 1, bfloat16 autocast, seed 0.
- At most 100 epochs; training stops after 5 epochs without a lower validation loss. The weights
  with the lowest validation loss are kept.

## Prediction

Detection and Detection+Identification (detector):

- The model runs over the whole chunk in 5-s windows with 50% overlap.
- The chunk is processed eight times, with its first 0, 1/8, ..., 7/8 of an output frame removed.
  The frames of the eight runs are interleaved in time order, which gives probabilities every
  16.4 ms (60.8 per second) while each value still comes from a full output frame of the model.
- Per taxon, runs of frames with probability of at least 0.5 become vocalizations. Of two
  vocalizations with an intersection over union above 0.8, across taxa, the one with the higher
  score is kept (the post-processing of the model repository, without its merging of
  neighbouring vocalizations).
- `detection`: all chunks of `prepared/test`, boundaries without taxa; chunks sampled above
  64 kHz get no predictions. `end_to_end`: all chunks of the subset, with taxa.

Identification (identifier): the vocalization is centred in 5 s of silence as in training; the
predicted taxon has the highest mean probability over the output frames the vocalization touches.

## Hardware and run time

| step | GPU | run time |
|---|---|---|
| detector, size 50 / 100 | RTX 3090 (24 GB) | 21 / 54 epochs, 5 / 17 min |
| detector, size 500 | RTX A6000 (48 GB) | 100 epochs, 50 min |
| Detection+Identification subset, size 500 | RTX A6000 | 20 min |
| detection on the whole test set, size 500 | RTX A6000 | 3.7 h |
| identifier, size 50 / 100 / 500 | RTX 3090 | 100 epochs each, 1.2 / 2.0 / 7.2 h |

Training needs about 14 GB of GPU memory. All training audio is held in memory at 32 kHz
(identifier at size 500: 16 GB).
