"""List the segmented xeno-canto vocalizations for pretraining.

    python -m systems.whisperid.pretraining.build_xc_manifest --data DIR --cap 200 --out FILE

Reads xeno_canto/segments.jsonl (one line per species: genus, species and its recordings
with the onsets and offsets of the detected vocalizations) and the audio in
xeno_canto/recordings. Recordings of the benchmark are left out, vocalizations between 0.02
and 5 s are kept, at most --cap per species.
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict

from benchmark import config
from systems.whisperid.pretraining import benchmark_recordings

SEED = 42
MIN_SECONDS = 0.02
MAX_SECONDS = 5.0


def vocalizations_by_species(directory, blocked):
    """Return species -> [(file name, onset, offset)] of the usable vocalizations."""
    available = set(os.listdir(directory / "recordings"))
    by_species = defaultdict(list)
    with open(directory / "segments.jsonl") as handle:
        for line in handle:
            entry = json.loads(line)
            species = f"{entry['genus']} {entry['species']}"
            for recording in entry.get("data", []):
                file_name = recording.get("file_name", "")
                if file_name not in available:
                    continue
                number = re.match(r"XC(\d+)", file_name)
                if number and number.group(1) in blocked:
                    continue
                segments = recording.get("segments") or {}
                onsets = segments.get("onset") or []
                offsets = segments.get("offset") or []
                for onset, offset in zip(onsets, offsets):
                    if MIN_SECONDS <= offset - onset <= MAX_SECONDS:
                        by_species[species].append((file_name, float(onset), float(offset)))
    return by_species


def main():
    """Write the manifest of xeno-canto vocalizations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config.use_data_root(args.data)

    directory = config.DATA / "xeno_canto"
    by_species = vocalizations_by_species(directory, benchmark_recordings.blocked_numbers())
    generator = random.Random(SEED)
    written = 0
    with open(args.out, "w") as handle:
        for species in sorted(by_species):
            vocalizations = by_species[species]
            if len(vocalizations) > args.cap:
                vocalizations = generator.sample(vocalizations, args.cap)
            for file_name, onset, offset in vocalizations:
                item = {
                    "audio": str(directory / "recordings" / file_name),
                    "species": species,
                    "onset": round(onset, 4),
                    "offset": round(offset, 4),
                }
                handle.write(json.dumps(item) + "\n")
                written += 1
    print(f"{written} vocalizations of {len(by_species)} species", flush=True)


if __name__ == "__main__":
    main()
