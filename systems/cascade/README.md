# Cascade: WhisperSeg, then WhisperID

The Detection+Identification cascade takes the vocalizations detected by the fine-tuned
WhisperSeg model and names each of them with the fine-tuned WhisperID model of the same
training size. Nothing is trained here.

## Files

| file | purpose |
|---|---|
| `predict.py` | names every WhisperSeg detection of the Detection+Identification chunks |

`predict.py` uses the spectrogram, marker, prompt and model loading of `systems/whisperid/` and
the spectrogram extractor of the WhisperSeg code set up in `systems/whisperseg/WhisperSeg`.

## Environment

Python 3.11, PyTorch 2.13.0, Transformers 4.44.2, numpy 2.4.6, librosa 0.11.0, soundfile 0.14.0,
pyarrow 25.0.0.

## Data

`--data` or the environment variable `CALLMARK_DATA` gives the data root (layout in
`benchmark/README.md`).

| input | written by |
|---|---|
| `predictions/whisperseg/detection_<size>.json.gz` | `python -m systems.whisperseg.predict --system whisperseg` |
| `models/whisperid/finetuned_<size>/` | fine-tuning of WhisperID (`systems/whisperid/README.md`) |
| `dataset/classification/train_<size>-*.parquet` | identification training examples: taxa and their frequencies |
| `prepared/test/<chunk_id>.wav` | audio of the test chunks |
| `labels/end_to_end_chunks.json` | chunks of the Detection+Identification subset |

## Command

Run from the repository root, once per training size (50, 100, 500).

```
export CALLMARK_DATA=/path/to/data
python -m systems.cascade.predict --size 50
```

It writes `predictions/cascade/end_to_end_<size>.json.gz`: for every chunk, the WhisperSeg
detections in their order, each with its boundaries unchanged and the taxon chosen by WhisperID.

## Method

| step | detail |
|---|---|
| window | 5 s of the chunk centred on the detection, moved to lie inside the chunk; a shorter chunk is padded with zeros |
| input | log-mel spectrogram of 500 columns (10 ms); the columns of the detection are marked in the four highest mel bands |
| time shifts | the window audio is rotated circularly by -0.25, 0 and +0.25 s and the marker moves with it; a shift other than 0 is used only if the shifted detection stays inside the window |
| candidate taxa | the 144 taxa of the examples WhisperID was fine-tuned on (training examples without its validation recordings) |
| taxon score | log-probability that the decoder continues the prompt `sr: <rate> kHz, species:` with the taxon name and the end-of-text token, computed for all taxa in one decoder pass over the prefix tree of the names |
| combination | softmax over the taxa per shift, mean over the used shifts, logarithm, minus 1.6 times the logarithm of the number of training examples of the taxon; the taxon with the highest value is the prediction |
| precision | 16-bit floating point, 12 windows per forward pass |

The model names about 12 detections per second on one RTX A6000:

| size | detections | run time |
|---|---|---|
| 50 | 52,583 | 1.2 h |
| 100 | 60,523 | 1.4 h |
| 500 | 56,408 | 1.3 h |
