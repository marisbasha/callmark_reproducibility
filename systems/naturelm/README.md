# NatureLM-audio

NatureLM-audio (`EarthSpeciesProject/NatureLM-audio`, an audio encoder connected to the
language model Llama 3.1 8B) with its LoRA adapters fine-tuned to answer with the taxon of a
vocalization.

## Environment

Python 3.11 with

| package | version |
|---|---|
| NatureLM-audio | 1.0.2, commit `c708df7a4cc294ca8d4aaf0498794b5674ce20b1` of https://github.com/earthspecies/NatureLM-audio |
| torch | 2.13.0 |
| transformers | 4.44.2 |
| peft | 0.11.1 |
| resampy | 0.4.3 |
| numpy | 2.4.6 |
| pyarrow | 25.0.0 |
| soundfile | 0.14.0 |

NatureLM-audio is installed as a wheel built from its repository, which leaves the choice of
the torch build to this environment:

    git clone https://github.com/earthspecies/NatureLM-audio.git
    cd NatureLM-audio
    git checkout c708df7a4cc294ca8d4aaf0498794b5674ce20b1
    uv build --wheel -o ..
    pip install ../naturelm_audio-1.0.2-py3-none-any.whl

The wheel does not contain the generation settings of the repository; copy them next to
`predict.py`:

    cp NatureLM-audio/configs/inference.yml systems/naturelm/inference.yml

The model weights are downloaded from the Hugging Face hub on first use; they require access
to `meta-llama/Meta-Llama-3.1-8B-Instruct`. Run from the repository root; `--data` can be
replaced by the environment variable `CALLMARK_DATA`. `clips.py` of `systems/frozen_encoders`
reads the examples, `common.py` of `systems/biolingual` holds the validation split and the
augmentation.

## Input

    dataset/classification/{train_50,train_100,train_500,test}-*.parquet

## Commands

    python -m systems.naturelm.train --data DIR --size 500      # also 50 and 100
    python -m systems.naturelm.predict --data DIR --size 500

| command | writes | run time on an RTX A6000 (48 GB) |
|---|---|---|
| `train` | `models/naturelm/train_<size>/` (the whole model, 16.6 GB) | 15, 23 and 78 min for sizes 50, 100 and 500 |
| `predict` | `predictions/naturelm/identification_<size>.json` | 66 min per size |

## Fine-tuning

| setting | value |
|---|---|
| input | the vocalization (onset to offset) at 16 kHz, resampled by linear interpolation; a vocalization shorter than 0.5 s is widened on both sides with the surrounding audio to 0.5 s |
| prompt | "What is the scientific name for the focal species in the audio?" |
| target | the taxon name |
| trained weights | the LoRA adapters of the language model (rank 32; 27.3 million parameters) |
| validation set | a tenth of the recordings of every taxon with at least two recordings, seed 42 |
| augmentation | as for BioLingual (`systems/biolingual/README.md`) |
| batch | 4, gradients accumulated over 8 batches |
| optimizer | AdamW, weight decay 0.01, gradient norm clipped to 1 |
| learning rate | one cycle over all steps, maximum 1e-4, 100 rising steps, cosine decay |
| steps | at most 8 epochs: 986, 1,595 and 3,914 steps for sizes 50, 100 and 500 |
| precision | bfloat16 autocast |
| validation | 12 times during the planned steps, loss on the first 40 validation batches |
| stopping | after 4 validations without a lower loss |
| kept model | the lowest validation loss |
| seed | 42 |

| training size | training examples | validation examples | steps run | lowest validation loss |
|---|---|---|---|---|
| 50 | 3,946 | 544 | 574 | 0.394 |
| 100 | 6,382 | 859 | 924 | 0.390 |
| 500 | 15,659 | 2,127 | 3,260 | 0.261 |

## Prediction

Every test vocalization is cut as for training, written to a 16-bit WAV file and given to the
inference pipeline of NatureLM-audio with the prompt above (beam search with 2 beams, the
generation settings of `inference.yml`). The answer is mapped to a taxon of the dataset: exact
match, then match ignoring case, then the most similar name with a similarity of at least 0.75
(`difflib`). An answer that matches no taxon is written as the empty string and counts as wrong
(482, 140 and 28 of 9,842 test examples for sizes 50, 100 and 500).
