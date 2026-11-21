# WhisperSeg

WhisperSeg (Gu et al. 2024) is a Whisper Large encoder-decoder that writes the onset and offset
of every vocalization of a 1,000-column spectrogram window as a token sequence. Two systems of
the benchmark are trained here, both for detection without taxa:

| system | starting point of fine-tuning |
|---|---|
| `whisperseg` | the released checkpoint after detection pretraining on synthetic recordings |
| `whisperseg_no_pretraining` | the released checkpoint |

## Base checkpoint

Both systems descend from the released multispecies checkpoint `nccratliri/whisperseg-large-ms`
(https://huggingface.co/nccratliri/whisperseg-large-ms). It is public and is downloaded from the
Hugging Face Hub when training starts; it is not trained by this repository.

For `whisperseg`, this checkpoint is first trained for one epoch on a synthetic detection set of
200 h (`build_pretraining_set.py`, `pretrain.py`). The detection-pretrained model is then
fine-tuned at each training size.

## Model code

The model, training and decoding code is https://github.com/nianlonggu/WhisperSeg at commit
`b9e79d43024ce86d1c5ab624d27dcede4cbbf169`, with the changes of `whisperseg.patch`. The upstream
repository carries no licence file, so its code is not copied here:

    git clone https://github.com/nianlonggu/WhisperSeg systems/whisperseg/WhisperSeg
    cd systems/whisperseg/WhisperSeg
    git checkout b9e79d43024ce86d1c5ab624d27dcede4cbbf169
    git apply ../whisperseg.patch

Changes of the patch:

- `train.py`, `--early_stop_patience`: training stops when the best validation score is that
  many validations old (checked in the second half of the planned steps). Upstream stopped after
  two consecutive decreases of the validation score.
- `train.py`, `model.py` (`save_model`): when old checkpoints are deleted, the checkpoint with
  the best validation score so far is kept, and only checkpoint folders are deleted.
- `train.py`, `--val_dataset_folder`: validation on a separate folder instead of a slice of
  every training file (used for pretraining).
- `train.py`: the selected checkpoint is not converted to CTranslate2 at the end of training;
  `convert.py` does this in the prediction environment (ctranslate2 4.8.1 with transformers
  4.44.2, the training environment, cannot convert the model).
- `datautils.py`: the zero padding in front of every training file has the data type of the
  audio (float32), and the unsliced audio is released. This lowers the memory needed for the
  pretraining set from about 300 GB to about 100 GB and does not change any value.
- `model.py` (`WhisperSegmenterFast`, the CTranslate2 decoder used for the test predictions):
  the decoder prompt ends with the species token with which every training target starts, and
  the end-of-text token is allowed directly after the prompt, so that a window without
  vocalizations can be decoded as empty.

The validation during training (`WhisperSegmenterForEval`) decodes with the upstream prompt.

## Environment

| step | packages |
|---|---|
| pretraining set | Python 3.11, numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, pyarrow 25.0.0, soundfile 0.14.0, soxr 1.1.0 |
| training | Python 3.11, torch 2.13.0, transformers 4.44.2, tokenizers 0.19.1, ctranslate2 4.8.1, numpy 2.4.6, librosa 0.11.0, scikit-learn 1.9.0, wandb 0.28.1 |
| conversion, prediction | Python 3.10, torch 2.5.1 (CUDA 12.1), transformers 4.38.2, tokenizers 0.15.2, ctranslate2 4.0.0 (needs cuDNN 8), numpy 1.26.4, scipy 1.15.3, scikit-learn 1.7.2, librosa 0.11.0, soundfile 0.13.1 |

## Data

    prepared/train_<size>_joined/        training files, written by
                                         python -m systems.data.join_windows (../data/README.md)
    prepared/test/                       test chunks (python -m systems.data.prepare_test)

Every annotation is trained as the single class `Vocal`; the taxa in the `cluster` field of the
training files are not used.

Only for the pretraining set:

    dataset/{classification,segmentation}/{train_500,test}-*.parquet
    labels/train_500_source_annotations.json   {chunk_id: [[onset, offset], ...]}: every annotation
                                               of the original corpora that overlaps a train_500
                                               chunk, in seconds from the start of the chunk. It
                                               includes the vocalizations that were removed when
                                               the dataset was built (longer than the 95th
                                               percentile of their corpus).
    birdset_xcl/XCL/XCL_metadata.parquet       metadata of BirdSet XCL
                                               (https://huggingface.co/datasets/DBD-research-group/BirdSet)
    birdset_xcl/audio/XC<number>.ogg           recordings extracted from the XCL archives: those
                                               of the bird taxa of the benchmark that were
                                               extracted for WhisperID pretraining (99,170
                                               recordings of 128 taxa, 32 kHz)

## Commands

Run from the repository root; `--data` can be replaced by the environment variable
`CALLMARK_DATA`. `<size>` is 50, 100 or 500, `<system>` is `whisperseg` or
`whisperseg_no_pretraining`.

    python -m systems.whisperseg.build_pretraining_set --data DIR
    python -m systems.whisperseg.pretrain --data DIR
    python -m systems.whisperseg.train --data DIR --system <system> --size <size>
    python -m systems.whisperseg.convert --data DIR --system <system> --size <size>
    python -m systems.whisperseg.predict --data DIR --system <system> --size <size>

| command | writes |
|---|---|
| `build_pretraining_set` | `prepared/whisperseg_pretraining/{train,val}/<corpus>__sr<rate>__<split><number>.wav`, `.json`; `prepared/whisperseg_pretraining/chunks/<chunk_id>.wav`, the train_500 chunks |
| `pretrain` | `models/whisperseg/pretraining/final_checkpoint` |
| `train` | `models/<system>/<size>/final_checkpoint` |
| `convert` | `models/<system>/<size>/final_checkpoint_ct2` (CTranslate2, `int8_float16`) |
| `predict` | `predictions/<system>/detection_<size>.json.gz` |

`build_pretraining_set` and `pretrain` are needed for `whisperseg` only.

## Pretraining set

Every file is a background of 30 to 60 s from one corpus and sampling rate with xeno-canto
vocalizations inserted; it carries the spectrogram settings of the corpus and one annotation
(`Vocal`) per inserted vocalization.

| part | rule |
|---|---|
| backgrounds | stretches of at least 0.75 s of the train_500 chunks without any annotation (dataset and `train_500_source_annotations.json`), 20 ms away from the adjacent annotations; chunks sampled above 48 kHz are not used |
| background of a file | random pieces of up to 20 s, a stretch chosen in proportion to its length, joined with 25-ms crossfades |
| hours per corpus and rate | proportional to the square root of its background hours, at most 25 times those hours |
| recordings | BirdSet XCL recordings in `birdset_xcl/audio`, without those whose xeno-canto number occurs in a recording identifier of the benchmark (training or test); 2 % of the recordings (by hash of the number) are used for `val/` only |
| events | the detected events of BirdSet outside its noise cluster, 0.02 to 5 s, at most 8 per recording, the same number for every taxon |
| boundaries | trimmed to the frames within 20 dB of the peak energy between 1 and 15 kHz (512-point spectrogram, 4-ms hop); an event is dropped if the 30 ms outside its boundaries reach within 6 dB of that peak |
| source noise | spectral gate with the mean spectrum of the quietest 20 % of the frames of the recording: gain max(0, 1 - 2 noise / power) |
| insertion | event with 10-ms margins and 5-ms fades, resampled to the rate of the background; 0.05 to 0.45 events per second; signal-to-background ratio uniform in -5 to 15 dB (background level in 5 s around the event); events 30 ms apart |
| overlap | with probability 0.2 an event is placed over an earlier event of another taxon, at -8 to 8 dB |
| seed | 20261030 |

| split | files | hours | vocalizations |
|---|---|---|---|
| train | 15,961 | 199.99 | 179,651 |
| val | 244 | 3.00 | 2,926 |

## Training

| setting | pretraining | fine-tuning |
|---|---|---|
| initial model | released checkpoint | `whisperseg`: pretrained model; `whisperseg_no_pretraining`: released checkpoint |
| training files | `prepared/whisperseg_pretraining/train` | `prepared/train_<size>_joined` |
| input | random 1,000-column window of a file (2.5 s; 0.5 s for the 250-kHz recordings) | the same |
| optimizer | AdamW, learning rate 3e-6, weight decay 0.01, 100 warm-up steps, linear decay, batch size 4, 16-bit floating point | the same |
| length | 1 epoch (77,818 steps) | at most 12 epochs |
| validation set | `prepared/whisperseg_pretraining/val` | 10 % cut from the start or the end (at random) of every training file; the rest of the file is trained on |
| validation | every 15,000 steps | after every epoch |
| early stopping | not reached | best score four validations old, from epoch 6 on |
| checkpoint | best validation score: mean of the interval F1 and the frame F1 of WhisperSeg's `evaluate`, one view per file | the same |
| seed | 66100 (numpy: validation slices, windows); the order of the batches is not seeded | the same |

Checkpoints of the stored predictions (RTX A6000, 48 GB):

| model | steps per epoch | epochs run | selected step (epoch) | run time |
|---|---|---|---|---|
| pretraining | 77,818 | 1 | 30,000 | 11.5 h |
| `whisperseg` 50 | 1,273 | 10 | 7,638 (6) | 2.5 h |
| `whisperseg` 100 | 2,058 | 10 | 12,348 (6) | 3.8 h |
| `whisperseg` 500 | 5,606 | 12 | 50,454 (9) | 12 h |
| `whisperseg_no_pretraining` 50 | 1,273 | 6 | 2,546 (2) | 1.5 h |
| `whisperseg_no_pretraining` 100 | 2,058 | 9 | 10,290 (5) | 3.5 h |
| `whisperseg_no_pretraining` 500 | 5,606 | 11 | 39,242 (7) | 10 h |

## Prediction

Each test chunk is decoded on its own with the CTranslate2 model (16-bit floating point, beam
size 4) and the spectrogram settings stored in its JSON file (`min_frequency`, `spec_time_step`,
`min_segment_length`, `eps`). Three views of the chunk, left-padded by 0, 1/3 and 2/3 of a
window, are decoded in batches of 8 windows, and their intervals are combined with WhisperSeg's
clustering rule (DBSCAN on the mean boundary distance, radius `eps`, at least 2 of 3 views); an
interval shorter than `min_segment_length` is dropped. The stored predictions were computed in
four parts on RTX A6000 and RTX 3090 cards, 40 min to 2.7 h per part and about 5 GB of GPU
memory.

Decoding in 16-bit floating point depends on the GPU model: on another card, boundaries can move
by a few milliseconds and single intervals can appear or disappear. WhisperSeg prints an error
raised while a chunk is decoded (for example when the GPU memory is exhausted) and returns the
chunk without vocalizations, so the output of `predict` has to be free of error messages.
