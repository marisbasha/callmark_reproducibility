# DualWhisper

DualWhisper is one WhisperSeg model that detects and identifies in a single decoding pass: for
every vocalization it emits the onset, the taxon and the offset. It is the WhisperSeg detector
of the same training size, trained further on the detection training set with the taxon of each
vocalization as its label. Given vocalizations are identified with the same model by fixing the
onset in the decoder prompt.

## Files

| file | purpose |
|---|---|
| `train.py` | training with validation after every epoch and checkpoint selection |
| `whisperseg_training.py` | started by `train.py`: WhisperSeg training script with the validation decoding of DualWhisper |
| `predict_end_to_end.py` | Detection+Identification on the chunks of `labels/end_to_end_chunks.json` |
| `predict_identification.py` | identification of the examples of `dataset/classification/test` |

## Model code

The scripts use the WhisperSeg code set up in `systems/whisperseg/WhisperSeg`
(https://github.com/nianlonggu/WhisperSeg, commit `b9e79d4`, with the changes described in
`systems/whisperseg/README.md`). DualWhisper relies on these changes of that code:

- In prediction with the CTranslate2 model, the decoder prompt ends with the species token
  that starts every training target, and the end-of-text token is allowed directly after the
  prompt, so that a window without vocalizations is decoded as empty.
- Training stops when the best validation score is `--early_stop_patience` validations old, and
  the best checkpoint is never removed when old checkpoints are deleted.

`whisperseg_training.py` starts the training script of that code and applies the same prompt
and end-of-text rule to the decoding of the validation set during training.

## Environment

| step | packages |
|---|---|
| training | Python 3.11, PyTorch 2.13.0, Transformers 4.44.2, CTranslate2 4.8.1, numpy 2.4.6, librosa 0.11.0 |
| prediction | Python 3.10, PyTorch 2.5.1, Transformers 4.38.2, CTranslate2 4.0.0 (requires cuDNN 8), numpy 1.26.4, librosa 0.11.0, soundfile 0.13.1, pyarrow 18.0.0 |

## Data

All scripts read one data root, given by `--data` or the environment variable `CALLMARK_DATA`
(layout in `benchmark/README.md`).

| input | written by |
|---|---|
| `prepared/train_<size>_joined/` | `python -m systems.data.join_windows` (`systems/data/README.md`): detection training windows joined into files of at least 30 s; `cluster` holds the taxon of each vocalization, `unknown` if unidentified |
| `models/whisperseg/<size>/final_checkpoint` | `python -m systems.whisperseg.train --system whisperseg` |
| `prepared/test/` | detection test chunks with their acoustic settings |
| `labels/end_to_end_chunks.json` | chunks of the Detection+Identification subset |
| `dataset/classification/test-*.parquet` | identification test examples |

## Commands

Run from the repository root, once per training size (50, 100, 500).

```
export CALLMARK_DATA=/path/to/data

python -m systems.dualwhisper.train --size 50
python -m systems.dualwhisper.predict_end_to_end --size 50
python -m systems.dualwhisper.predict_identification --size 50
```

| command | writes |
|---|---|
| `train` | `models/dualwhisper/<size>/final_checkpoint` (Hugging Face format) and `final_checkpoint_ct2` (CTranslate2, `int8_float16`) |
| `predict_end_to_end` | `predictions/dualwhisper/end_to_end_<size>.json.gz` |
| `predict_identification` | `predictions/dualwhisper/identification_<size>.json` |

At the end of training the WhisperSeg script converts the selected checkpoint with the command
`python`. The conversion works in the prediction environment and fails in the training
environment (CTranslate2 4.8.1 with Transformers 4.44.2), without stopping the script. Either
put the `python` of the prediction environment first on the search path during training, or
convert afterwards in the prediction environment:

```
python systems/whisperseg/WhisperSeg/convert_hf_to_ct2.py --quantization int8_float16 \
    --model $CALLMARK_DATA/models/dualwhisper/50/final_checkpoint \
    --output_dir $CALLMARK_DATA/models/dualwhisper/50/final_checkpoint_ct2
```

## Training

| setting | value |
|---|---|
| initial model | WhisperSeg detector of the same training size (detection pretraining, then fine-tuning) |
| labels | 144 taxa and `unknown`; the label table of the initial model is cleared |
| input | windows of 1,000 spectrogram columns, batch size 4 |
| optimizer | AdamW, learning rate 3e-6, weight decay 0.01, 100 warm-up steps, linear decay over 12 epochs, mixed precision |
| validation set | 10 % cut from the start or the end (at random) of every training file; the rest of the file is trained on |
| validation | after every epoch: mean of the interval F1 and the frame F1 of WhisperSeg's `evaluate`, with the taxon required to agree |
| stopping | after 12 epochs, or from epoch 6 on when the best validation score is four epochs old |
| selection | checkpoint with the highest validation score |
| seed | 66100 (numpy: validation slices, windows); the order of the batches is not seeded |

| size | steps per epoch | epochs run | selected step (epoch) | run time (one RTX A6000, 48 GB) |
|---|---|---|---|---|
| 50 | 1,273 | 11 | 8,911 (7) | 2.2 h |
| 100 | 2,058 | 12 | 24,696 (12) | 3.9 h |
| 500 | 5,606 | 12 | 67,272 (12) | 10.5 h |

## Prediction

Detection+Identification: every chunk is decoded with the CTranslate2 model (16-bit floating
point, beam size 4, batches of 8 windows) and the settings stored in its `prepared/test/*.json`
(`min_frequency`, `spec_time_step`, `min_segment_length`, `eps`). Three views of the chunk,
left-padded by 0, 1/3 and 2/3 of a window, are decoded and combined with WhisperSeg's
clustering rule; a vocalization keeps the taxon decoded for it. The 1,315 chunks were decoded
in four parts of about 30 min each on RTX 3090 cards.

Identification: the model window starts two time tokens before the given onset. The decoder
prompt is the training prefix followed by the time token of the onset. Beam search with five
hypotheses; the prediction is the taxon of the most probable hypothesis that names one,
otherwise `unknown`. The 9,842 examples take about 6 min on one RTX A6000.

Decoding uses 16-bit floating point on the GPU; boundaries can differ by a few milliseconds
between GPU models.
