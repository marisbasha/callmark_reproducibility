"""Replace the backgrounds of a manifest that come from validation recordings.

    python -m systems.whisperid.pretraining.redraw_backgrounds --data DIR --manifest FILE --out FILE

A background stretch from a recording that is held out for validation at any training size
(identification or detection) is replaced by a random stretch of the same corpus from the
remaining recordings.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark import config, data
from systems.whisperid import examples

SEED = 42
VALIDATION_FRACTION = 0.1
BACKGROUND_KEYS = ("bg_audio", "bg_start", "bg_end", "bg_dataset")


def detection_validation_recordings(chunks):
    """Return the recordings held out for validation of the detectors: a tenth per corpus."""
    recordings_of_corpus = defaultdict(set)
    for chunk in chunks:
        recordings_of_corpus[chunk["dataset"]].add(chunk["rec_id"])
    held_out = set()
    for corpus in sorted(recordings_of_corpus):
        recordings = sorted(recordings_of_corpus[corpus])
        if len(recordings) < 2:
            continue
        digest = hashlib.blake2b(f"{SEED}:{corpus}".encode(), digest_size=4).digest()
        generator = np.random.RandomState(int.from_bytes(digest, "big") % (2 ** 31))
        generator.shuffle(recordings)
        count = int(round(VALIDATION_FRACTION * len(recordings)))
        count = min(max(1, count), len(recordings) - 1)
        held_out.update(recordings[:count])
    return held_out


def validation_recordings():
    """Return the recordings held out for validation at any training size."""
    held_out = set()
    for size in config.SIZES:
        rows = examples.rows(f"train_{size}", with_audio=False)
        _, validation = examples.split_validation(rows)
        for example in validation:
            held_out.add(example["rec_id"])
        chunks = data.table("segmentation", f"train_{size}", ("rec_id",))
        held_out.update(detection_validation_recordings(chunks))
    return held_out


def main():
    """Write the manifest with redrawn backgrounds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config.use_data_root(args.data)

    recording_of_chunk = {}
    for chunk in data.table("segmentation", "train_500", ("chunk_id", "rec_id")):
        recording_of_chunk[chunk["chunk_id"]] = chunk["rec_id"]
    held_out = validation_recordings()

    with open(args.manifest) as handle:
        items = [json.loads(line) for line in handle]
    allowed = defaultdict(set)
    for item in items:
        if recording_of_chunk[Path(item["bg_audio"]).stem] not in held_out:
            background = tuple(item[key] for key in BACKGROUND_KEYS)
            allowed[item["bg_dataset"]].add(background)
    allowed = {corpus: sorted(backgrounds) for corpus, backgrounds in allowed.items()}

    redrawn = 0
    with open(args.out, "w") as handle:
        for index, item in enumerate(items):
            if recording_of_chunk[Path(item["bg_audio"]).stem] in held_out:
                generator = random.Random(SEED * 1_000_003 + index)
                background = generator.choice(allowed[item["bg_dataset"]])
                item.update(dict(zip(BACKGROUND_KEYS, background)))
                redrawn += 1
            handle.write(json.dumps(item) + "\n")
    print(f"{redrawn} of {len(items)} backgrounds redrawn", flush=True)


if __name__ == "__main__":
    main()
