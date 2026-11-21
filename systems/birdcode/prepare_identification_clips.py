"""Write the identification training examples of one training size as 5-s clips.

    python -m systems.birdcode.prepare_identification_clips --data DIR --size 50

Every example of dataset/classification/train_<size> becomes one clip with the vocalization
centred in silence, in the format of the prepared detection training windows:

    birdcode/identification_clips_<size>/clips/<corpus>__<row index>.wav, .json
    birdcode/identification_clips_<size>/validation.txt     clips held out for model selection
"""
import argparse
import io
import json
import random
from collections import defaultdict

import pyarrow.parquet as pq
import soundfile

from benchmark import config, data
from systems.birdcode import birdcode

VALIDATION_FRACTION = 0.1
VALIDATION_SEED = 42
STORED_SUBTYPES = ("PCM_16", "PCM_24", "PCM_32", "FLOAT")


def read_examples(size):
    """Return the training examples without audio, each with the name of its clip."""
    columns = ("species", "rec_id", "sampling_rate", "onset_in_window", "offset_in_window")
    examples = data.table("classification", f"train_{size}", columns)
    for index, example in enumerate(examples):
        example["clip"] = f"{example['dataset']}__{index:06d}"
    return examples


def validation_examples(examples):
    """Return the examples held out for model selection.

    For every taxon with at least two recordings, a tenth of its recordings (at least one,
    never all) is held out with all their examples."""
    generator = random.Random(VALIDATION_SEED)
    by_taxon = defaultdict(lambda: defaultdict(list))
    for example in examples:
        by_taxon[example["species"]][example["rec_id"]].append(example)
    held_out = []
    for by_recording in by_taxon.values():
        recordings = sorted(by_recording)
        generator.shuffle(recordings)
        if len(recordings) < 2:
            continue
        count = int(round(VALIDATION_FRACTION * len(recordings)))
        count = min(max(1, count), len(recordings) - 1)
        for recording in recordings[:count]:
            held_out.extend(by_recording[recording])
    return held_out


def audio_of_examples(size):
    """Yield the encoded audio of every training example in the order of read_examples."""
    files = sorted((config.dataset_dir() / "classification").glob(f"train_{size}-*.parquet"))
    for file in files:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=64, columns=["audio", "dataset"]):
            for row in batch.to_pylist():
                if row["dataset"] != config.DUPLICATE_CORPUS:
                    yield row["audio"]["bytes"]


def write_clip(directory, example, audio_bytes):
    """Write the clip of one example and its annotation file."""
    buffer = io.BytesIO(audio_bytes)
    subtype = soundfile.info(buffer).subtype
    if subtype not in STORED_SUBTYPES:
        subtype = "FLOAT"
    buffer.seek(0)
    waveform, sampling_rate = soundfile.read(buffer, dtype="float32")
    clip, onset, offset = birdcode.centre_vocalization(
        waveform,
        sampling_rate,
        float(example["onset_in_window"]),
        float(example["offset_in_window"]),
    )
    soundfile.write(str(directory / f"{example['clip']}.wav"), clip, sampling_rate, subtype=subtype)
    annotation = {
        "sr": sampling_rate,
        "onset": [onset],
        "offset": [offset],
        "cluster": [example["species"]],
    }
    (directory / f"{example['clip']}.json").write_text(json.dumps(annotation))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data root (default: environment variable CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    args = parser.parse_args()
    config.use_data_root(args.data)

    examples = read_examples(args.size)
    held_out = validation_examples(examples)
    out_dir = birdcode.identification_clip_dir(args.size)
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    held_out_clips = sorted(example["clip"] for example in held_out)
    (out_dir / "validation.txt").write_text("\n".join(held_out_clips) + "\n")
    print(f"{len(examples)} examples, {len(held_out)} held out", flush=True)

    for example, audio_bytes in zip(examples, audio_of_examples(args.size)):
        write_clip(clip_dir, example, audio_bytes)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
