"""Choose the xeno-canto recordings that are segmented: at most 50 per species.

    python -m systems.sources.xeno_canto_recordings --data DIR

Reads xeno_canto/metadata.jsonl and the audio files in xeno_canto/recordings/ and writes
xeno_canto/selected_recordings.jsonl: one line per species, {"genus", "species", "data":
[{"file_name"}, ...]}.

The recordings of a species that exist in xeno_canto/recordings/ are shuffled. A fifth of them
(at least one) is set aside and of the others the first 50 are taken. The only recording of a
species with a single recording is taken.
"""
import argparse
import json
import os

import numpy
from mutagen import File

from benchmark import config

SEED = 100
MAX_RECORDINGS = 50
SET_ASIDE = 0.2


def available_recordings(directory):
    """Return [(genus, species, [file name, ...])] for the species with audio files."""
    species_list = []
    with open(directory / "metadata.jsonl") as handle:
        for line in handle:
            entry = json.loads(line)
            file_names = []
            for record in entry["data"]:
                name = record["file_original_name"]
                if not isinstance(name, str) or name == "":
                    continue
                if os.path.exists(directory / "recordings" / name):
                    file_names.append(name)
            if file_names:
                species_list.append((entry["genus"], entry["species"], file_names))
    return species_list


def main():
    """Write the selected recordings of every species."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    directory = config.DATA / "xeno_canto"
    species_list = available_recordings(directory)
    numpy.random.seed(SEED)
    for _, _, file_names in species_list:
        numpy.random.shuffle(file_names)

    with open(directory / "selected_recordings.jsonl", "w") as handle:
        for genus, species, file_names in species_list:
            if len(file_names) == 1:
                # A recording of length zero is not used.
                if File(directory / "recordings" / file_names[0]).info.length == 0:
                    continue
                selected = file_names
            else:
                set_aside = max(1, int(numpy.round(len(file_names) * SET_ASIDE)))
                selected = file_names[:min(MAX_RECORDINGS, len(file_names) - set_aside)]
            line = {"genus": genus, "species": species,
                    "data": [{"file_name": name} for name in selected]}
            handle.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
