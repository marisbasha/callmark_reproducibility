"""Write the Voxaboxen input of the identification model for one training size.

    python -m systems.voxaboxen.prepare_identification --data DIR --size 50

Every identification example is cut to its vocalization and centred in 3 s of silence, with the
vocalization and its taxon as the only annotation:

    voxaboxen/identification_<size>/train_audio/train_<row>.wav, train_tables/train_<row>.txt
    voxaboxen/identification_<size>/test_audio/test_<row>.wav, test_tables/test_<row>.txt
    voxaboxen/identification_<size>/train_info.csv, val_info.csv, test_info.csv
    voxaboxen/identification_<size>/test_examples.json    name, identifier and position of the
                                                          vocalization of every test clip
"""
import argparse
import io
import random
from collections import defaultdict

import numpy
import pyarrow.parquet as pq
import soundfile

from benchmark import config, data
from systems.voxaboxen import files

CLIP_SECONDS = 3.0
MARGIN_SECONDS = 0.5
ULTRASONIC_RATE = 200000
VALIDATION_FRACTION = 0.1
VALIDATION_SEED = 42
COLUMNS = ["audio", "sampling_rate", "species", "dataset", "chunk_id", "rec_id", "event_index",
           "onset_in_window", "offset_in_window"]


def examples_of(split):
    """Yield the identification examples of one split in dataset order, with their audio."""
    paths = sorted((config.dataset_dir() / "classification").glob(f"{split}-*.parquet"))
    for path in paths:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            for example in parquet.read_row_group(group, columns=COLUMNS).to_pylist():
                if example["dataset"] != config.DUPLICATE_CORPUS:
                    yield example


def padded_clip(example):
    """Return the vocalization centred in silence, the written rate and its position in seconds.

    Recordings at 200 kHz and above are slowed by writing a lower sampling rate. A
    vocalization of 3 s or more gets 0.5 s of silence on both sides."""
    waveform, _ = soundfile.read(io.BytesIO(example["audio"]["bytes"]), dtype="float32")
    rate = int(example["sampling_rate"])
    written_rate = rate
    if rate >= ULTRASONIC_RATE:
        written_rate = rate // files.power_of_two_slow_down(rate)
    first = max(0, int(example["onset_in_window"] * rate))
    last = min(len(waveform), int(example["offset_in_window"] * rate))
    vocalization = waveform[first:last]
    length = int(CLIP_SECONDS * written_rate)
    if len(vocalization) >= length:
        length = len(vocalization) + 2 * int(MARGIN_SECONDS * written_rate)
    left = (length - len(vocalization)) // 2
    clip = numpy.zeros(length, dtype="float32")
    clip[left:left + len(vocalization)] = vocalization
    onset = left / written_rate
    offset = (left + len(vocalization)) / written_rate
    return clip, written_rate, onset, offset


def write_clips(split, prefix, target_dir):
    """Write the clip and selection table of every example of a split; return their records."""
    audio_dir = target_dir / f"{prefix}_audio"
    table_dir = target_dir / f"{prefix}_tables"
    audio_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, example in enumerate(examples_of(split)):
        name = f"{prefix}_{index:06d}"
        clip, written_rate, onset, offset = padded_clip(example)
        audio_path = audio_dir / f"{name}.wav"
        table_path = table_dir / f"{name}.txt"
        soundfile.write(str(audio_path), clip, written_rate)
        files.write_table(table_path, [(onset, offset, example["species"])])
        records.append({
            "name": name,
            "example_id": data.example_id(example),
            "species": example["species"],
            "rec_id": example["rec_id"],
            "onset": onset,
            "offset": offset,
            "row": (name, str(audio_path.resolve()), str(table_path.resolve())),
        })
    return records


def split_training_records(records):
    """Return (training, validation) records.

    For every taxon with at least two recordings, a tenth of its recordings (at least one,
    never all) is held out with all their examples."""
    generator = random.Random(VALIDATION_SEED)
    by_taxon = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_taxon[record["species"]][record["rec_id"]].append(record)
    training = []
    validation = []
    for by_recording in by_taxon.values():
        recordings = sorted(by_recording)
        generator.shuffle(recordings)
        count = 0
        if len(recordings) >= 2:
            count = int(round(VALIDATION_FRACTION * len(recordings)))
            count = min(max(1, count), len(recordings) - 1)
        for recording in recordings[:count]:
            validation.extend(by_recording[recording])
        for recording in recordings[count:]:
            training.extend(by_recording[recording])
    return training, validation


def rows_of(records):
    """Return the file-list rows of clip records."""
    return [record["row"] for record in records]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="", help="data root (default: CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    target_dir = files.input_dir("identification", arguments.size)
    training, validation = split_training_records(
        write_clips(f"train_{arguments.size}", "train", target_dir))
    files.write_info(target_dir / "train_info.csv", rows_of(training))
    files.write_info(target_dir / "val_info.csv", rows_of(validation))

    test = write_clips("test", "test", target_dir)
    files.write_info(target_dir / "test_info.csv", rows_of(test))
    examples = []
    for record in test:
        examples.append({key: record[key] for key in ("name", "example_id", "onset", "offset")})
    data.write_json(target_dir / "test_examples.json", examples)
    print(f"{len(training)} training clips, {len(validation)} validation clips, "
          f"{len(test)} test clips -> {target_dir}")


if __name__ == "__main__":
    main()
