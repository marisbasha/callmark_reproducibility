"""List the xeno-canto recordings of the VoCallBase collection.

    python -m systems.sources.xeno_canto_metadata --data DIR

Writes xeno_canto/metadata.jsonl: one line per species, {"genus", "species", "data": [record of
one recording, ...]}. The field file_original_name of a record is the name of its audio file.
"""
import argparse
import json

import requests

from benchmark import config

API = "https://dataplatform.evolvinglanguage.ch/animal_call"


def get_json(path, parameters=None):
    """Return the JSON answer of one request to the collection."""
    response = requests.get(f"{API}/json/{path}", params=parameters)
    response.raise_for_status()
    return response.json()


def main():
    """Write the records of all recordings, species by species."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    directory = config.DATA / "xeno_canto"
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "metadata.jsonl", "w") as handle:
        for genus in get_json("list_genus/"):
            for entry in get_json(f"list_species/{genus}"):
                species = entry["species"]
                records = get_json("data", {"genus": genus, "species": species})
                line = {"genus": genus, "species": species, "data": records}
                handle.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
