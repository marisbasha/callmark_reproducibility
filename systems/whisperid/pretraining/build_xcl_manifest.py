"""List the BirdSet XCL vocalizations of the benchmark bird taxa for pretraining.

    python -m systems.whisperid.pretraining.build_xcl_manifest --data DIR --out FILE

Reads birdset_xcl/XCL (XCL_metadata.parquet and the audio shards of the XCL configuration of
BirdSet) and birdset_xcl/ebird_taxonomy.csv. For every bird taxon of the identification
training set, the detected events (0.02 to 5 s) of all its recordings outside the benchmark
are taken in turn over the recordings, at most 20,000 per taxon. The audio of the used
recordings is extracted to birdset_xcl/audio.
"""
import argparse
import csv
import json
import random
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas

from benchmark import config
from systems.whisperid import examples
from systems.whisperid.pretraining import benchmark_recordings

SEED = 42
MIN_SECONDS = 0.02
MAX_SECONDS = 5.0
MAX_PER_TAXON = 20000
EXTRACTION_PROCESSES = 6


def scientific_names(taxonomy_file):
    """Return eBird species code -> scientific name."""
    names = {}
    with open(taxonomy_file) as handle:
        for row in csv.DictReader(handle):
            names[row["SPECIES_CODE"]] = row["SCIENTIFIC_NAME"]
    return names


def bird_taxa():
    """Return the bird taxa of the identification training set."""
    taxa = set()
    for example in examples.rows("train_500", with_audio=False):
        if example["taxon"] == "bird":
            taxa.add(example["species"])
    return taxa


def usable_recordings(directory):
    """Return the XCL recordings of the benchmark bird taxa that are not in the benchmark."""
    columns = ["filepath", "ebird_code", "detected_events"]
    recordings = pandas.read_parquet(directory / "XCL" / "XCL_metadata.parquet", columns=columns)
    names = scientific_names(directory / "ebird_taxonomy.csv")
    recordings["species"] = recordings.ebird_code.map(names)
    recordings = recordings[recordings.species.isin(bird_taxa())]
    blocked = {int(number) for number in benchmark_recordings.blocked_numbers()}
    recordings["number"] = recordings.filepath.str.extract(r"XC(\d+)")[0].astype(int)
    return recordings[~recordings.number.isin(blocked)]


def events_of_recordings(recordings):
    """Return, per recording in random order, its [(file path, onset, offset)] events."""
    recordings = recordings.sample(frac=1, random_state=SEED)
    per_recording = []
    for file_path, detected in zip(recordings.filepath, recordings.detected_events):
        events = []
        if detected is not None:
            for event in detected:
                onset = float(event[0])
                offset = float(event[1])
                if MIN_SECONDS <= offset - onset <= MAX_SECONDS:
                    events.append((file_path, onset, offset))
        per_recording.append(events)
    return per_recording


def take_in_turn(per_recording, generator):
    """Return up to MAX_PER_TAXON events, one random event per recording in each round."""
    taken = []
    while len(taken) < MAX_PER_TAXON and any(per_recording):
        for events in per_recording:
            if events and len(taken) < MAX_PER_TAXON:
                taken.append(events.pop(generator.randrange(len(events))))
    return taken


def extract(task):
    """Extract the wanted files of one shard that are not extracted yet."""
    shard, wanted, audio_directory = task
    with tarfile.open(shard, "r:gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and name in wanted and not (audio_directory / name).exists():
                with archive.extractfile(member) as source:
                    (audio_directory / name).write_bytes(source.read())
    return shard.name


def main():
    """Write the manifest and extract the audio it refers to."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config.use_data_root(args.data)

    directory = config.DATA / "birdset_xcl"
    audio_directory = directory / "audio"
    generator = random.Random(SEED)
    items = []
    for species, recordings in usable_recordings(directory).groupby("species"):
        per_recording = events_of_recordings(recordings)
        for file_path, onset, offset in take_in_turn(per_recording, generator):
            items.append({
                "audio": str(audio_directory / Path(file_path).name),
                "species": species,
                "onset": onset,
                "offset": offset,
            })

    wanted = {Path(item["audio"]).name for item in items}
    audio_directory.mkdir(parents=True, exist_ok=True)
    shards = sorted((directory / "XCL").glob("XCL_shard_*.tar.gz"))
    tasks = [(shard, wanted, audio_directory) for shard in shards]
    with ProcessPoolExecutor(EXTRACTION_PROCESSES) as pool:
        for shard_name in pool.map(extract, tasks):
            print(shard_name, flush=True)

    written = 0
    with open(args.out, "w") as handle:
        for item in items:
            if Path(item["audio"]).exists():
                handle.write(json.dumps(item) + "\n")
                written += 1
    print(f"{written} vocalizations", flush=True)


if __name__ == "__main__":
    main()
