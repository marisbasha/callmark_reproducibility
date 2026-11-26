# MC-WhisperSeg: vocalization detection with body-mounted sensors

Code for the section "Body-mounted sensor channels support caller attribution"
(Fig. 4, Extended Data figure on missed and overlapping vocalizations,
Supplementary Table "Matched sensor comparison"). Three systems are trained on
recordings of pairs of zebra finches and tested on held-out pairs and on public
recordings of two, four and eight birds. The microphone-only detector is then
trained again with group audio labelled by MC-WhisperSeg:

| system | input | output |
|---|---|---|
| `microphone` | the shared microphone | vocalizations of all birds |
| `accelerometer` | the accelerometer of the focal bird | vocalizations of the focal bird |
| `multichannel` (MC-WhisperSeg) | focal accelerometer, summed accelerometers of the other birds, microphone | vocalizations of the focal bird |

## Files

| file | purpose |
|---|---|
| `data.py` | data root, loading and normalization of recordings, reference vocalizations |
| `train.py` | training with validation and checkpoint selection |
| `infer.py` | inference on the validation and test sets |
| `score.py` | caller-independent detection, caller-specific detection, overlap analyses, statistics |
| `annotators.py` | the two manual annotations of the public recordings scored against the consolidated reference |
| `label_groups.py` | labels unannotated group recordings with MC-WhisperSeg |
| `added_audio_sets.py`, `added_audio_sets.json` | training sets of pair recordings plus 7.3, 29.9 or 91.1 h of labelled group audio |
| `added_audio.py` | all systems and both manual annotations on the group recordings: detection F1, frame F1, detection by number of simultaneously vocalizing birds, F1 by boundary tolerance |
| `mcwhisperseg/` | model code, modified from MCWhisperSeg (Apache-2.0) |

## Data

All scripts read one data root, given by the environment variable `CALLMARK_DATA`
or by `--data`:

```
$CALLMARK_DATA/multichannel/
    train/             88 chunks of pair recordings, once per focal bird:
                       <chunk>_channel_<bird>_as_target_channel_1.wav   focal accelerometer
                       <chunk>_channel_<bird>_as_target_channel_2.wav   accelerometer of the other bird
                       <chunk>_channel_<bird>_as_target_channel_3.wav   microphone
                       <chunk>_channel_<bird>_as_target_channel_1.json  vocalizations of the focal bird
    val/               14 chunks from separate recording sessions, same format
    train_microphone/  <chunk>_as_target_channel_1.wav (microphone) and .json (vocalizations of both birds)
    val_microphone/    same format
    test_lab/          16 held-out recordings of pairs: <recording>.wav (accelerometer 1, accelerometer 2,
                       microphone) and <recording>.json (onset, offset, channelIndex)
    test_public/       BirdPark recordings from Zenodo (10.5281/zenodo.13144875): segments.h5 and
                       <experiment>/<recording>.h5
    models/<system>/   ct2 (CTranslate2 model used for inference), checkpoint, validation.json
    predictions/<system>/<set>/<recording>.json
    results/           scores.json, annotators.json, added_audio.json
    group_audio/       unannotated group recordings labelled by label_groups.py, one chunk per recording:
                       <experiment>__<recording>_as_target_channel_1.wav (microphone) and .json
                       (vocalizations of all birds)

$CALLMARK_DATA/added_audio/<set>/multichannel/
                       data root of the microphone-only detector trained with added group audio
                       (<set>: 7h, 30h, 91h), written by added_audio_sets.py
```

Training, validation and `test_lab` audio is sampled at 16 kHz. Training and validation
waveforms are stored after the normalization in `data.normalize` has been applied per
recording; `test_lab` recordings are normalized on loading. The public recordings are
stored at 24,414.0625 Hz; `data.load_public` high-pass filters their accelerometer
channels at 250 Hz, normalizes and resamples them.

## Experimental fixed mel-power floor

Set `CALLMARK_ABSOLUTE_MEL=1` for both training and prediction to use
a fixed floor of 1e-10 in mel power. This retains the original log10 transform
and affine scaling while removing the per-window 80 dB cutoff. With consistent
waveform scaling, a distant click can no longer raise the floor under a quiet
call in the same window. Input gain still matters. Leave the variable unset to
reproduce the published checkpoint preprocessing.

## Environment

Python 3.10, PyTorch 2.5.1, Transformers 4.38.2, CTranslate2 4.0.0 (requires cuDNN 8),
librosa 0.11.0, h5py, soundfile, scipy, scikit-learn, matplotlib, ipywidgets, pandas.
Scoring needs only numpy, scipy, h5py, soundfile and librosa.

## Reproduction

```
export CALLMARK_DATA=/path/to/data

# training, one GPU per system; selects the checkpoint with the highest validation F1
python train.py --system microphone
python train.py --system accelerometer
python train.py --system multichannel

# inference
for system in microphone accelerometer multichannel; do
    python infer.py --system $system --set test_lab
    python infer.py --system $system --set test_public
done

# results
python score.py
python annotators.py
```

`score.py` prints and writes to `results/scores.json`:

- `detection`: caller-independent F1 per cohort with exact paired randomization tests
  across recordings and Holm-adjusted values (Fig. 4c, top of the table);
- `attribution`: caller-specific precision, recall and F1 of the accelerometer and
  multichannel systems with a paired bootstrap over recordings (bottom of the table);
- `overlap`: share of overlapping vocalizations per group size (Fig. 4b), vocalizations
  detected by number of simultaneously vocalizing birds, bootstrap tests of the detection
  F1 (Fig. 4c), and missed vocalizations, overlapping vocalizations, overlapping pairs and
  overlap duration (Extended Data figure).

`annotators.py` writes `results/annotators.json` (hatched bars in Fig. 4c and the values
quoted in the table caption).

### Microphone-only detector trained on group audio labelled by MC-WhisperSeg (Fig. 4d, e)

```
# label unannotated group recordings (<experiment>/.../<recording>.h5) with MC-WhisperSeg
python label_groups.py /path/to/group_recordings

# training sets: the pair recordings plus the group recordings listed in added_audio_sets.json
python added_audio_sets.py

# train and run the microphone-only detector on each set
for set in 7h 30h 91h; do
    python train.py --system microphone --data $CALLMARK_DATA/added_audio/$set
    python infer.py --system microphone --set test_public --data $CALLMARK_DATA/added_audio/$set
done

# results
python added_audio.py
```

`added_audio_sets.json` lists the 63, 257 and 783 group recordings of the three sets, all from
days without a test recording. Recordings in which a bird carried no transmitter or a transmitter
had failed are left out, because such a bird is heard on the microphone without being labelled.

`added_audio.py` prints and writes to `results/added_audio.json`, for the 14 recordings of four
and eight birds: detection F1 against the hours of training audio (Fig. 4d), the share of
vocalizations found when one, two, or three and more birds vocalize at the same time (Fig. 4e),
the frame F1 and the detection F1 at boundary tolerances of 5 to 200 ms.

Bootstrap tests use fixed seeds; training is seeded but not bit-reproducible on GPU.

## Model code

`mcwhisperseg/` contains `model.py`, `datautils.py`, `audio_utils.py`, `utils.py` and
`convert_hf_to_ct2.py` from https://github.com/nianlonggu/MCWhisperSeg (commit d9e7e12),
distributed under the Apache License 2.0 (`mcwhisperseg/LICENSE`). `audio_utils.py`,
`utils.py` and `convert_hf_to_ct2.py` are unchanged. Changes in `model.py` and `datautils.py`:

- The log-mel blocks of all input channels are stacked to 80 x channels input bands.
  Upstream reduced the stack to 80 bands by keeping every third row.
- `load_model` takes `num_channels` and widens the first convolution of the encoder to
  accept the stacked input. The filters of the first block keep the pretrained weights,
  those of the other blocks start at zero.
- The decoder prompt ends with the species token that starts every training target, and
  the end-of-text token is allowed directly after the prompt, so that a window without
  vocalizations can be decoded as empty. Special tokens are kept when the output of the
  Hugging Face model is decoded.
- Errors in data-loading and generation threads are raised instead of being ignored.

The upstream training script, the segmentation service, notebooks and documentation
are not included; training is done by `train.py`.
