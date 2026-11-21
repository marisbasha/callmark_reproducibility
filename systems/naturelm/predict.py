"""Identification predictions of the fine-tuned NatureLM-audio model.

    python -m systems.naturelm.predict --data DIR --size 500

Reads models/naturelm/train_<size>/ and writes predictions/naturelm/identification_<size>.json.
The model is asked PROMPT for every test vocalization; its answer is mapped to a taxon of the
dataset, and to the empty string if no taxon is close to it.
"""
import argparse
import difflib
import re
import tempfile
from pathlib import Path

import soundfile
from NatureLM.infer import Pipeline
from NatureLM.models import NatureLM

from benchmark import config, data
from systems.biolingual import common
from systems.frozen_encoders import clips
from systems.naturelm.train import MINIMUM_SECONDS, PROMPT, SAMPLING_RATE

# The generation settings of the NatureLM-audio repository (configs/inference.yml).
GENERATION_SETTINGS = Path(__file__).parent / "inference.yml"
# The pipeline starts every answer with the time span it refers to: "#0.00s - 0.70s#: ".
TIME_SPAN = re.compile(r"^\s*#[^#]*#\s*:?\s*")
SIMILARITY = 0.75


def taxon_of(answer, taxon_names, by_lower_case):
    """Return the taxon named by an answer of the model, or '' if none is close enough.

    Exact match first, then match ignoring case, then the most similar taxon name."""
    name = TIME_SPAN.sub("", answer.strip())
    name = name.split("\n")[0].strip().strip(".")
    if name in taxon_names:
        return name
    if name.lower() in by_lower_case:
        return by_lower_case[name.lower()]
    closest = difflib.get_close_matches(name, taxon_names, n=1, cutoff=SIMILARITY)
    if closest:
        return closest[0]
    return ""


def predict(size, examples):
    """Return the predicted taxon of every example, in the given order."""
    taxon_names = common.taxa()
    by_lower_case = {name.lower(): name for name in taxon_names}
    model_directory = config.DATA / "models" / "naturelm" / f"train_{size}"
    model = NatureLM.from_pretrained(str(model_directory)).eval().cuda()
    # The cache of audio encodings is keyed by a sample of the waveform, so two different
    # clips can share an entry; every clip is encoded once here, so the cache is switched off.
    model.audio_encoding_cache = None
    pipeline = Pipeline(model=model, cfg_path=str(GENERATION_SETTINGS))
    predicted = []
    with tempfile.TemporaryDirectory() as directory:
        clip_file = str(Path(directory) / "clip.wav")
        for index, example in enumerate(examples):
            samples = clips.vocalization(example, SAMPLING_RATE, MINIMUM_SECONDS)
            # The pipeline reads the clip from a 16-bit WAV file.
            soundfile.write(clip_file, samples, SAMPLING_RATE)
            duration = len(samples) / SAMPLING_RATE
            answers = pipeline(
                [clip_file],
                [PROMPT],
                window_length_seconds=duration,
                hop_length_seconds=duration,
            )
            predicted.append(taxon_of(answers[0], taxon_names, by_lower_case))
            if index % 100 == 0:
                print(f"{index}/{len(examples)}", flush=True)
    return predicted


def main():
    """Predict the test examples with the model of one training size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    args = parser.parse_args()
    config.use_data_root(args.data)
    examples = clips.examples("test")
    predicted = predict(args.size, examples)
    example_ids = [data.example_id(example) for example in examples]
    output_file = config.prediction_file("naturelm", "identification", args.size)
    data.write_json(output_file, dict(zip(example_ids, predicted)))
    print(output_file)


if __name__ == "__main__":
    main()
