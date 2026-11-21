# WhisperID

WhisperID identifies the taxon of a marked vocalization. A Whisper encoder-decoder (initialized from
WhisperSeg) reads a 5-s log-mel spectrogram in which the vocalization is marked in the four highest mel
bands, and generates the scientific name after the prompt `sr: <kHz> kHz, species:`. It is pretrained on
xeno-canto vocalizations mixed into benchmark training audio and fine-tuned on each training size.

Run every command from the repository root. `--data` can be replaced by the environment variable
`CALLMARK_DATA` (data root of `benchmark/README.md`).

## Environment

| package | version |
|---|---|
| python | 3.11.14 |
| torch | 2.13.0 (CUDA 13.0) |
| transformers | 4.44.2 |
| numpy | 2.4.6 |
| librosa | 0.11.0 |
| soundfile | 0.14.0 |
| pyarrow, pandas | 25.0.0, 2.3.3 |

- The spectrogram extractor is `WhisperSegFeatureExtractor` of `audio_utils.py` from
  https://github.com/nianlonggu/WhisperSeg (commit b9e79d4), unchanged. It is imported from the clone in
  `systems/whisperseg/WhisperSeg` (see `systems/whisperseg/README.md`).
- The encoded audio of a split is held in memory while it is used (6 GB for `train_500`, 4 GB for
  `test`).
- If TensorFlow is installed in the same environment, set `USE_TF=0`: `transformers` otherwise loads it
  with the spectrogram extractor, and loading it before parts of PyTorch can crash the process.

## Files

| file | content |
|---|---|
| `examples.py` | identification examples, audio, validation split |
| `features.py` | spectrogram, marker, augmentation, prompt |
| `model.py` | model loading (250 encoder frames), training tokens, greedy generation |
| `finetune.py` | fine-tuning on one training size |
| `predict.py` | test predictions |
| `embed.py` | frozen-encoder embeddings for the probes |
| `pretraining/` | manifests of xeno-canto vocalizations and the pretraining stages |

## Fine-tuning and prediction

    python -m systems.whisperid.finetune --data DIR --size 50      # also 100, 500
    python -m systems.whisperid.predict --data DIR --size 50

| step | reads | writes |
|---|---|---|
| `finetune` | `models/whisperid/pretrained`, `dataset/classification/train_<size>` | `models/whisperid/finetuned_<size>` |
| `predict` | `models/whisperid/finetuned_<size>`, `dataset/classification/test` | `predictions/whisperid/identification_<size>.json` |

Fine-tuning:

- Validation set: for every taxon, a tenth of its recordings (at least one, none for a taxon with a single
  recording), seed 42: 544, 859 and 2,127 examples; 3,946, 6,382 and 15,659 examples remain for training.
- All weights, AdamW, learning rate 1e-5, batch size 8, bfloat16 autocast, gradient norm clipped at 1,
  seed 42, natural taxon frequencies. Every batch holds one sampling rate; the sampling rate of the next
  batch is drawn at random.
- Augmentation, redrawn in every epoch: gain of -6 to 6 dB, Gaussian noise at 20 to 40 dB signal-to-noise
  ratio, random polarity, circular shift by up to a tenth of the audio (the marker moves with it).
- After every epoch the validation examples are decoded greedily; training stops when their macro-F1 has
  not improved for 8 epochs (at most 50 epochs; the stored runs stopped after 14, 22 and 23 epochs).
- The model is the average of the weights after every epoch from the second epoch on.

Prediction:

- Every taxon of the training examples (144) is scored by the log-probability of its name followed by the
  end token, with the model in half precision.
- The scores are turned into probabilities over the taxa and averaged over the circular shifts 0, -0.25
  and +0.25 s of the audio; a shift is used when the shifted vocalization stays inside the audio.
- 1.6 times the log training frequency of the taxon is subtracted from the log mean probability; the taxon
  with the highest value is the prediction.

## Embeddings for the probes

    python -m systems.whisperid.embed --data DIR --encoder whisperid_pretrained

Writes `embeddings/<encoder>.npy` (float32, 27,628 x 1,280): the encoder output of the marked spectrogram
in single precision, averaged over the encoder frames of the vocalization, for all examples of `train_500`
and then of `test`.

| encoder | checkpoint |
|---|---|
| `whisperseg` | `nccratliri/whisperseg-large-ms` |
| `whisper_large` | `openai/whisper-large` |
| `xeno_canto_only` | `models/whisperid/xeno_canto` |
| `whisperid_bridge` | `models/whisperid/bridge` |
| `whisperid_bridge_xcl1` | `models/whisperid/bridge_xcl` |
| `whisperid_pretrained` | `models/whisperid/pretrained` |
| `bridge_replay` | `models/whisperid/bridge_replay` |

## Pretraining

The fine-tuned models start from `models/whisperid/pretrained`, the last of four stages:

| stage | starts from | manifest | epochs | batch | validation fraction |
|---|---|---|---|---|---|
| `bridge_first` | `nccratliri/whisperseg-large-ms` | `xc_bridge.jsonl` | 1 | 16 | 0.005 |
| `bridge` | `bridge_first` | `xc_bridge.jsonl` | 2 | 8 | 0.005 |
| `bridge_xcl` | `bridge` | `xcl_bridge.jsonl` | 1 | 8 | 0.005 |
| `pretrained` | `bridge_xcl` | `xcl_bridge_redrawn.jsonl`, `--mixture-seed 43` | 1 | 8 | 0.005 |

Two further checkpoints are used only as probe encoders:

| stage | starts from | manifest | epochs | batch | validation fraction |
|---|---|---|---|---|---|
| `xeno_canto` | `nccratliri/whisperseg-large-ms` | `xc_200.jsonl` | 4 | 8 | 0.01 |
| `bridge_replay` | `nccratliri/whisperseg-large-ms` | `xc_bridge_replay.jsonl` | 2 | 16 | 0.005 |

Every stage is one call, for example

    python -m systems.whisperid.pretraining.pretrain --manifest DIR/whisperid/manifests/xcl_bridge_redrawn.jsonl \
        --base DIR/models/whisperid/bridge_xcl --out DIR/models/whisperid/pretrained \
        --epochs 1 --batch 8 --validation-fraction 0.005 --mixture-seed 43

- Input: 5-s window at 32 kHz. A vocalization with a background is cut to at most 4 s, placed at a random
  position in a background stretch (repeated when shorter than 5 s) and scaled to -5 to 15 dB relative to
  the background; a second vocalization of another taxon, if listed, is added at -8 to 8 dB; the mixture is
  scaled down when it exceeds full scale. A vocalization without background is shown in its own recording,
  the window centred on it and moved by up to 1 s.
- Objective, optimizer and precision as in fine-tuning. Whole recordings are held out for validation
  (seed 42); the epoch with the lowest validation loss is kept.

### Data

| folder of the data root | content | source |
|---|---|---|
| `xeno_canto/recordings/` | xeno-canto recordings | VoCallBase collection of xeno-canto recordings (https://dataplatform.evolvinglanguage.ch/animal_call) |
| `xeno_canto/segments.jsonl` | one line per species: `{"genus", "species", "data": [{"file_name", "segments": {"onset": [...], "offset": [...]}}]}`, times in seconds | vocalizations detected in every recording with WhisperSeg |
| `birdset_xcl/XCL/` | `XCL_metadata.parquet`, `XCL_shard_*.tar.gz` | configuration XCL of BirdSet (Hugging Face `DBD-research-group/BirdSet`) |
| `birdset_xcl/ebird_taxonomy.csv` | species rows of the eBird taxonomy (`SPECIES_CODE`, `SCIENTIFIC_NAME`) | eBird/Clements taxonomy |
| `birdset_xcl/audio/` | recordings extracted by `build_xcl_manifest` | |
| `whisperid/background_chunks/` | written by `export_background_chunks` | |
| `whisperid/manifests/` | the manifests below | |

A manifest is a JSON-lines file; every line holds `audio`, `species`, `onset`, `offset` (seconds in the
recording) and, for a mixture, `bg_audio`, `bg_start`, `bg_end`, `bg_dataset` and optionally
`overlay_audio`, `overlay_onset`, `overlay_offset`.

### Manifests

With `M=DIR/whisperid/manifests` and `P="python -m systems.whisperid.pretraining"`:

    $P.export_background_chunks --data DIR
    $P.build_xc_manifest --data DIR --cap 200 --out $M/xc_200.jsonl
    $P.build_xc_manifest --data DIR --cap 1000000 --out $M/xc_all.jsonl
    $P.build_bridge_manifest --data DIR --vocalizations $M/xc_all.jsonl --cap 1000 --out $M/xc_bridge.jsonl
    $P.build_replay_manifest --mixtures $M/xc_bridge.jsonl --vocalizations $M/xc_200.jsonl --out $M/xc_bridge_replay.jsonl
    $P.build_xcl_manifest --data DIR --out $M/xcl.jsonl
    $P.build_bridge_manifest --data DIR --vocalizations $M/xcl.jsonl --cap 20000 --out $M/xcl_bridge.jsonl
    $P.redraw_backgrounds --data DIR --manifest $M/xcl_bridge.jsonl --out $M/xcl_bridge_redrawn.jsonl

| script | content of the manifest | size in the stored runs |
|---|---|---|
| `build_xc_manifest` | segmented xeno-canto vocalizations of 0.02 to 5 s, at most `--cap` per species | 185,212 vocalizations of 1,079 species at cap 200 |
| `build_xcl_manifest` | detected events of BirdSet XCL for the bird taxa of the identification training set, at most 20,000 per taxon, taken in turn over the recordings | 994,479 vocalizations of 128 taxa, 99,170 recordings |
| `build_bridge_manifest` | every vocalization (at most `--cap` per taxon) with a random background stretch; half of them with a second vocalization, kept when its taxon differs | 657,214 mixtures (xeno-canto), 994,479 (XCL); 10,779 stretches, 11.3 h |
| `build_replay_manifest` | the mixtures plus half as many vocalizations without background | 985,821 lines |
| `redraw_backgrounds` | backgrounds from recordings held out for validation replaced by stretches of other recordings of the same corpus | |

- Recordings whose xeno-canto number occurs in a recording name of the benchmark are left out of all
  manifests.
- Background stretches are the parts without annotated vocalization, at least 0.75 s long, of the 500-size
  detection training chunks of `enabirds`, `sne_full`, `nbm` and `wytham`.

### Stored checkpoints

The pretraining audio of the stored runs is not part of the published dataset, and their background chunks
came from an earlier build of the training split with the same rules. The stored manifest of the last
stage held 994,381 mixtures (989,410 for training) on 531 background chunks, none from a test or validation
recording. Of the stored runs, the third of four epochs was kept for `xeno_canto`, the second of two for
`bridge`, the first of two for `bridge_replay`.

## Hardware and run time

One NVIDIA RTX A6000 (48 GB); times of the stored runs.

| step | time | GPU memory |
|---|---|---|
| pretraining stage `pretrained` (123,676 steps) | 14.6 h | 36 GB |
| fine-tuning, sizes 50, 100, 500 | 1.2 h, 2.9 h, 7.0 h | 33 GB |
| prediction, per size | 0.8 h | 15 GB |
| embeddings, per encoder | 12 min | 10 GB |
