"""Write the Voxaboxen input of the detection model or of the taxon model for one training size.

    python -m systems.voxaboxen.prepare_detection --data DIR --size 50 --model detection
    python -m systems.voxaboxen.prepare_detection --data DIR --size 50 --model taxon

The detection model is trained with the single label "Vocal", the taxon model with the taxon of
every vocalization. Both read the joined training windows prepared/train_<size>_joined
(python -m systems.data.join_windows) and the prepared test set, and write

    voxaboxen/<model>_<size>/selection_tables/<file>.txt    onset, offset and label per file
    voxaboxen/<model>_<size>/audio/<file>.wav               slowed copies of ultrasonic files
    voxaboxen/<model>_<size>/train_info.csv, val_info.csv, test_info.csv
    voxaboxen/<model>_<size>/test_slow_down.json            slow-down factor of every test chunk
"""
import argparse
import hashlib
from collections import Counter, defaultdict

import numpy
import soundfile

from benchmark import config, data
from systems.voxaboxen import files

CLIP_SECONDS = 10.0
CLIP_HOP_SECONDS = 5.0
VALIDATION_FRACTION = 0.1
VALIDATION_SEED = 42
UNIDENTIFIED = "unknown"


def prepared_files(directory):
    """Return the label files of a prepared folder in name order."""
    return sorted(directory.glob("*.json"))


# ---------------------------------------------------------------- labels
def vocal_annotations(directory):
    """Return file name -> [(onset, offset, "Vocal")] for a prepared folder."""
    annotations = {}
    for path in prepared_files(directory):
        labels = data.read_json(path)
        vocalizations = []
        for onset, offset in zip(labels["onset"], labels["offset"]):
            vocalizations.append((onset, offset, "Vocal"))
        annotations[path.stem] = vocalizations
    return annotations


def taxon_training_annotations(directory, size):
    """Return file name -> [(onset, offset, taxon)] for the joined training files.

    Over the files in name order, the first `size` vocalizations of a taxon keep their taxon;
    later ones and unidentified ones are labelled "Unknown", which Voxaboxen trains for
    detection only."""
    named = Counter()
    annotations = {}
    for path in sorted(prepared_files(directory), key=stem_of):
        labels = data.read_json(path)
        vocalizations = []
        for onset, offset, taxon in zip(labels["onset"], labels["offset"], labels["cluster"]):
            if taxon != UNIDENTIFIED and named[taxon] < size:
                named[taxon] += 1
                vocalizations.append((onset, offset, taxon))
            else:
                vocalizations.append((onset, offset, "Unknown"))
        annotations[path.stem] = vocalizations
    return annotations


def stem_of(path):
    """Return the file name without its extension."""
    return path.stem


def taxon_test_annotations():
    """Return chunk -> [(onset, offset, taxon)] of the test set, "Unknown" if unidentified."""
    reference = data.dataset_reference()
    taxa = data.test_taxa()
    annotations = {}
    for chunk_id, vocalizations in reference.items():
        labelled = []
        for (onset, offset), taxon in zip(vocalizations, taxa[chunk_id]):
            labelled.append((onset, offset, taxon or "Unknown"))
        annotations[chunk_id] = labelled
    return annotations


# ---------------------------------------------------------------- Voxaboxen files
def slow_down_factor(rate, min_frequency):
    """Return the factor by which a file is slowed so that its band fits below 8 kHz.

    Only the ultrasonic corpora declare a minimum frequency; their audio is written with the
    sampling rate in the header divided by the factor."""
    if not min_frequency or rate <= 2 * files.MODEL_RATE:
        return 1
    return files.power_of_two_slow_down(rate)


def write_voxaboxen_files(source_dir, annotations, target_dir):
    """Write the selection table of every file of `source_dir` and slowed audio where needed.

    Returns the rows of the file list and the slow-down factor of every file."""
    audio_dir = target_dir / "audio"
    table_dir = target_dir / "selection_tables"
    audio_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    factors = {}
    for path in prepared_files(source_dir):
        name = path.stem
        labels = data.read_json(path)
        rate = int(labels["sr"])
        factor = slow_down_factor(rate, labels.get("min_frequency", 0))
        factors[name] = factor
        audio_path = path.with_suffix(".wav")
        if factor > 1:
            waveform, _ = soundfile.read(str(audio_path), dtype="float32")
            audio_path = audio_dir / f"{name}.wav"
            soundfile.write(str(audio_path), waveform, rate // factor)
        slowed = []
        for onset, offset, label in annotations[name]:
            slowed.append((onset * factor, offset * factor, label))
        table_path = table_dir / f"{name}.txt"
        files.write_table(table_path, slowed)
        rows.append((name, str(audio_path.resolve()), str(table_path.resolve())))
    return rows, factors


# ---------------------------------------------------------------- training and validation lists
def validation_names(rows):
    """Return the names of the joined files held out for validation.

    A tenth of the files of every corpus (at least one, never all) is drawn with a generator
    seeded from the corpus name; a corpus with a single file is not split."""
    by_corpus = defaultdict(list)
    for name, _, _ in rows:
        by_corpus[data.corpus_of(name)].append(name)
    held_out = set()
    for corpus in sorted(by_corpus):
        names = sorted(by_corpus[corpus])
        if len(names) < 2:
            continue
        digest = hashlib.blake2b(f"{VALIDATION_SEED}:{corpus}".encode(), digest_size=4).digest()
        generator = numpy.random.RandomState(int.from_bytes(digest, "big") % (2 ** 31))
        generator.shuffle(names)
        count = int(round(VALIDATION_FRACTION * len(names)))
        count = min(max(1, count), len(names) - 1)
        held_out.update(names[:count])
    return held_out


def training_clips(seconds):
    """Return the number of clips the Voxaboxen training loader cuts from a file."""
    return max(0, int(numpy.floor((seconds - CLIP_SECONDS) // CLIP_HOP_SECONDS)))


def header_rate(row):
    """Return the sampling rate in the header of the audio file of a file-list row."""
    return soundfile.info(row[1]).samplerate


def rows_within_clip_budget(rows, factors):
    """Return the rows of the files that are used, ordered by corpus, header rate and name.

    Files without a single training clip (shorter than 15 s) are left out. The budget of a
    corpus is the number of clips of its files at their original speed; files are taken in
    order until the clips of the files taken reach the budget."""
    by_corpus = defaultdict(list)
    for row in rows:
        by_corpus[data.corpus_of(row[0])].append(row)
    kept = []
    for corpus in sorted(by_corpus):
        corpus_rows = sorted(by_corpus[corpus], key=header_rate)
        budget = 0
        for name, audio_path, _ in corpus_rows:
            seconds = soundfile.info(audio_path).duration
            budget += training_clips(seconds / factors.get(name, 1))
        clips = 0
        for row in corpus_rows:
            clips_of_file = training_clips(soundfile.info(row[1]).duration)
            if clips_of_file == 0:
                continue
            if clips >= budget:
                break
            kept.append(row)
            clips += clips_of_file
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="", help="data root (default: CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    parser.add_argument("--model", required=True, choices=("detection", "taxon"))
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)
    size = arguments.size

    joined_dir = config.prepared_dir(f"train_{size}_joined")
    test_dir = config.prepared_dir("test")
    if arguments.model == "detection":
        training_annotations = vocal_annotations(joined_dir)
        test_annotations = vocal_annotations(test_dir)
    else:
        training_annotations = taxon_training_annotations(joined_dir, size)
        test_annotations = taxon_test_annotations()

    target_dir = files.input_dir(arguments.model, size)
    rows, factors = write_voxaboxen_files(joined_dir, training_annotations, target_dir)
    held_out = validation_names(rows)
    training_rows = []
    validation_rows = []
    for row in rows:
        if row[0] in held_out:
            validation_rows.append(row)
        else:
            training_rows.append(row)
    # The slow-down factors enter the budget of the training list only: every validation file
    # with at least one clip is used.
    training_rows = rows_within_clip_budget(training_rows, factors)
    validation_rows = rows_within_clip_budget(validation_rows, {})
    files.write_info(target_dir / "train_info.csv", training_rows)
    files.write_info(target_dir / "val_info.csv", validation_rows)

    test_rows, test_factors = write_voxaboxen_files(test_dir, test_annotations, target_dir)
    files.write_info(target_dir / "test_info.csv", test_rows)
    data.write_json(target_dir / "test_slow_down.json", test_factors)
    print(f"{len(training_rows)} training files, {len(validation_rows)} validation files, "
          f"{len(test_rows)} test files -> {target_dir}")


if __name__ == "__main__":
    main()
