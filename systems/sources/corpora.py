"""Readers of the original corpora of the benchmark.

Every corpus lies in its own folder corpora/<name>/ of the data root (README.md gives the origin
and the layout of each). A reader returns one dictionary per annotated recording:

    {"corpus", "audio" (path of the audio file), "rec_id" (<corpus>__<file stem>),
     "events": [(onset, offset, taxon), ...]}

Times are seconds from the start of the audio file. The taxon is a scientific name; it is None
for a vocalization whose label names no taxon of the benchmark.
"""
import argparse
import csv
import glob
import os
from pathlib import Path

import pandas
import soundfile

from benchmark import config

# Corpus -> taxonomic group of its vocalizations.
GROUP = {
    "budgie": "bird",
    "canary": "bird",
    "enabirds": "bird",
    "infant_marmosets": "primate",
    "nbm": "bird",
    "ozf": "bird",
    "right_whale": "cetacean",
    "rodent_usv": "mammal",
    "rodents2019": "mammal",
    "singing_mice": "mammal",
    "sne_full": "bird",
    "spiders": "arachnid",
    "tworadiobirds": "bird",
    "wytham": "bird",
}

# Mammals annotated in the bird soundscapes.
GROUP_OF_TAXON = {
    "Tamias striatus": "mammal",
    "Tamiasciurus hudsonicus": "mammal",
    "Odocoileus virginianus": "mammal",
}

# Value of the column dataset_name.
CORPUS_LABEL = {
    "budgie": "Budgerigar",
    "canary": "Domestic canary",
    "enabirds": "Eastern NA birds",
    "infant_marmosets": "Infant marmosets",
    "nbm": "NBM nocturnal birds",
    "ozf": "OZF zebra finch",
    "right_whale": "Right whale",
    "rodent_usv": "Rodent USV",
    "rodents2019": "Rodents 2019",
    "singing_mice": "Singing mice",
    "sne_full": "Sierra Nevada (full)",
    "spiders": "Spiders (JS)",
    "tworadiobirds": "TwoRadioBirds",
    "wytham": "Wytham great tit",
}

# Corpora with a single taxon.
TAXON_OF_CORPUS = {
    "budgie": "Melopsittacus undulatus",
    "canary": "Serinus canaria",
    "infant_marmosets": "Callithrix jacchus",
    "ozf": "Taeniopygia castanotis",
    "right_whale": "Eubalaena glacialis",
    "singing_mice": "Scotinomys teguina",
    "spiders": "Araneae sp.",
    "tworadiobirds": "Taeniopygia castanotis",
    "wytham": "Parus major",
}

# Folder of the rodent recordings -> taxon. Strains and pups of the house mouse are one taxon.
RODENT_TAXON = {
    "gerbil": "Meriones unguiculatus",
    "rat_pleasant": "Rattus norvegicus",
    "rat_distressed": "Rattus norvegicus",
    "mouse_B6pup": "Mus musculus",
    "mouse_BALBc": "Mus musculus",
    "mouse_C57BL": "Mus musculus",
    "mouse_Shank2": "Mus musculus",
}

# Four-letter codes of the Eastern North American recordings that are taxa of the benchmark.
# A vocalization with any other code is kept without taxon.
ENABIRDS_TAXON = {
    "AMCR": "Corvus brachyrhynchos",
    "AMGO": "Spinus tristis",
    "AMRE": "Setophaga ruticilla",
    "AMRO": "Turdus migratorius",
    "BCCH": "Poecile atricapillus",
    "BGGN": "Polioptila caerulea",
    "BHCO": "Molothrus ater",
    "BHVI": "Vireo solitarius",
    "BLJA": "Cyanocitta cristata",
    "BTNW": "Setophaga virens",
    "CARW": "Thryothorus ludovicianus",
    "CEDW": "Bombycilla cedrorum",
    "COYE": "Geothlypis trichas",
    "CSWA": "Setophaga pensylvanica",
    "EATO": "Pipilo erythrophthalmus",
    "HETH": "Catharus guttatus",
    "HOWA": "Setophaga citrina",
    "NOCA": "Cardinalis cardinalis",
    "NOFL": "Colaptes auratus",
    "OVEN": "Seiurus aurocapilla",
    "RBGR": "Pheucticus ludovicianus",
    "RBWO": "Melanerpes carolinus",
    "REVI": "Vireo olivaceus",
    "SCTA": "Piranga olivacea",
    "SWTH": "Catharus ustulatus",
    "TUTI": "Baeolophus bicolor",
    "VEER": "Catharus fuscescens",
    "WBNU": "Sitta carolinensis",
    "WITU": "Meleagris gallopavo",
    "WOTH": "Hylocichla mustelina",
}

# Labels of the nocturnal-migration corpus that are renamed; "Other" is not used.
NBM_RENAMED = {
    "Philloscopus collybita": "Phylloscopus collybita",
    "Sylvia communis": "Curruca communis",
    "Other": None,
}
NBM_SUBSETS = ("train_nbm_xc", "test")

# Call types of the marmoset corpus that are not vocalizations: 11 silence, 12 noise.
MARMOSET_NOT_VOCAL = (11, 12)


def corpus_dir(corpus):
    """Return the folder of one original corpus."""
    return config.DATA / "corpora" / corpus


def recording(corpus, audio, stem, events):
    """Return the dictionary of one recording."""
    return {"corpus": corpus, "audio": str(audio), "rec_id": f"{corpus}__{stem}", "events": events}


def sorted_files(directory, pattern):
    """Return the sorted paths (as strings) matching `pattern` below `directory`."""
    return sorted(glob.glob(str(Path(directory) / pattern), recursive=True))


# ---------------------------------------------------------------- budgerigars, singing mice
def read_table_per_recording(corpus, pattern, separator):
    """Read recordings with one table next to each WAV file, columns onset and offset.

    A table that cannot be read is skipped."""
    taxon = TAXON_OF_CORPUS[corpus]
    recordings = []
    for table_path in sorted_files(corpus_dir(corpus), pattern):
        audio = table_path[:-4] + ".wav"
        if not os.path.exists(audio):
            continue
        try:
            table = pandas.read_csv(table_path, sep=separator)
            events = []
            for onset, offset in zip(table["onset"], table["offset"]):
                events.append((float(onset), float(offset), taxon))
        except Exception:
            continue
        if events:
            recordings.append(recording(corpus, audio, Path(audio).stem, events))
    return recordings


def read_budgie():
    """Budgerigars: a comma-separated table next to every WAV file, in any subfolder."""
    return read_table_per_recording("budgie", "**/*.csv", ",")


def read_singing_mice():
    """Singing mice: a semicolon-separated table next to every WAV file."""
    return read_table_per_recording("singing_mice", "*.csv", ";")


# ---------------------------------------------------------------- zebra finch group
def read_ozf():
    """Zebra finch group: one Raven selection table per FLAC file; every row is a vocalization."""
    root = corpus_dir("ozf")
    recordings = []
    for table_path in sorted_files(root, "**/selection_tables/*.txt"):
        stem = Path(table_path).stem
        audio = Path(table_path).parent.parent / "audio" / (stem + ".flac")
        if not audio.exists():
            continue
        events = []
        with open(table_path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                onset = float(row["Begin Time (s)"])
                offset = float(row["End Time (s)"])
                events.append((onset, offset, TAXON_OF_CORPUS["ozf"]))
        if events:
            recordings.append(recording("ozf", audio, stem, events))
    return recordings


# ---------------------------------------------------------------- rodents
def read_onset_offset_rows(table_path):
    """Return the (onset, offset) pairs of a table without header; other rows are skipped."""
    pairs = []
    with open(table_path) as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                pairs.append((float(row[0]), float(row[1])))
            except ValueError:
                continue
    return pairs


def read_rodent_usv():
    """Rodent ultrasonic vocalizations: one folder per group of animals, a table per WAV file."""
    recordings = []
    for folder in sorted_files(corpus_dir("rodent_usv"), "*"):
        taxon = RODENT_TAXON.get(os.path.basename(folder))
        if not taxon:
            continue
        for audio in sorted_files(folder, "*.wav"):
            table_path = audio[:-4] + ".csv"
            if not os.path.exists(table_path):
                continue
            events = []
            for onset, offset in read_onset_offset_rows(table_path):
                events.append((onset, offset, taxon))
            if events:
                recordings.append(recording("rodent_usv", audio, Path(audio).stem, events))
    return recordings


def read_rodents2019():
    """The recordings of rodent_usv a second time, every one labelled as house mouse.

    This corpus is ignored wherever the dataset is read. It is part of the build because its
    house-mouse vocalizations share the split and the caps with those of rodent_usv."""
    recordings = []
    for folder in sorted_files(corpus_dir("rodent_usv"), "*"):
        for audio in sorted_files(folder, "*.wav"):
            table_path = audio[:-4] + ".csv"
            if not os.path.exists(table_path):
                continue
            events = []
            for onset, offset in read_onset_offset_rows(table_path):
                if offset > onset:
                    events.append((onset, offset, "Mus musculus"))
            if events:
                recordings.append(recording("rodents2019", audio, Path(audio).stem, events))
    return recordings


# ---------------------------------------------------------------- nocturnal bird migration
def read_nbm():
    """Nocturnal bird migration: an Audacity label file next to every WAV file.

    Label lines (onset, offset, name) alternate with frequency lines that start with a
    backslash. The subset train_nbm_orig is not part of the benchmark."""
    recordings = []
    for subset in NBM_SUBSETS:
        for audio in sorted_files(corpus_dir("nbm") / subset, "*.wav"):
            label_path = audio[:-4] + ".txt"
            if not os.path.exists(label_path):
                continue
            events = []
            with open(label_path) as handle:
                for line in handle:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 3 or fields[0] == "\\" or not fields[2].strip():
                        continue
                    name = fields[2].strip()
                    taxon = NBM_RENAMED.get(name, name)
                    if not taxon:
                        continue
                    try:
                        events.append((float(fields[0]), float(fields[1]), taxon))
                    except ValueError:
                        continue
            if events:
                stem = f"{subset}__{Path(audio).stem}"
                recordings.append(recording("nbm", audio, stem, events))
    return recordings


# ---------------------------------------------------------------- Eastern North American birds
def read_enabirds():
    """Eastern North American birds: a Raven selection table next to every WAV file."""
    recordings = []
    for table_path in sorted_files(corpus_dir("enabirds"), "*/*.selections.txt"):
        table_path = Path(table_path)
        audio = table_path.parent / (table_path.name.split(".Table")[0] + ".wav")
        if not audio.exists():
            continue
        events = []
        with open(table_path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                try:
                    code = (row.get("Species") or "").strip().upper()
                    taxon = ENABIRDS_TAXON.get(code)
                    events.append((float(row["Begin Time (s)"]), float(row["End Time (s)"]), taxon))
                except (KeyError, ValueError):
                    continue
        if events:
            recordings.append(recording("enabirds", audio, audio.stem, events))
    return recordings


# ---------------------------------------------------------------- spiders, right whales
def read_few_shot_detection_corpus(corpus):
    """Read a subset of FASD13: a table (Starttime, Endtime, Q) per WAV file; Q = POS is an
    event."""
    taxon = TAXON_OF_CORPUS[corpus]
    recordings = []
    for table_path in sorted_files(corpus_dir(corpus), "*.csv"):
        audio = Path(table_path).with_suffix(".wav")
        if not audio.exists():
            continue
        events = []
        with open(table_path) as handle:
            for row in csv.DictReader(handle):
                if str(row.get("Q", "POS")).strip().upper() != "POS":
                    continue
                try:
                    events.append((float(row["Starttime"]), float(row["Endtime"]), taxon))
                except (KeyError, ValueError):
                    continue
        if events:
            recordings.append(recording(corpus, audio, audio.stem, events))
    return recordings


def read_spiders():
    """Jumping spiders (subset JS of FASD13)."""
    return read_few_shot_detection_corpus("spiders")


def read_right_whale():
    """Right whales (subset RW of FASD13)."""
    return read_few_shot_detection_corpus("right_whale")


# ---------------------------------------------------------------- canary
def read_canary():
    """Domestic canary: an Audacity label file (start, end, syllable) per WAV file.

    The label SIL marks silence."""
    root = corpus_dir("canary")
    audio_of_stem = {}
    for path in root.rglob("*.wav"):
        audio_of_stem[path.stem] = path
    recordings = []
    for label_path in sorted(root.rglob("*.audacity.txt")):
        stem = label_path.name.replace(".audacity.txt", "")
        audio = audio_of_stem.get(stem)
        if audio is None:
            continue
        events = []
        with open(label_path) as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) < 3 or row[2].strip().upper() == "SIL":
                    continue
                try:
                    onset = float(row[0])
                    offset = float(row[1])
                except ValueError:
                    continue
                if offset > onset:
                    events.append((onset, offset, TAXON_OF_CORPUS["canary"]))
        if events:
            recordings.append(recording("canary", audio, stem, events))
    return recordings


# ---------------------------------------------------------------- infant marmosets
def read_infant_marmosets():
    """Infant marmosets: labels.csv (filename, start, end, duration, calltype, caller).

    Rows of the call types silence and noise are not vocalizations."""
    root = corpus_dir("infant_marmosets")
    audio_of_name = {}
    for path in root.rglob("*.wav"):
        audio_of_name.setdefault(path.name, path)
        audio_of_name.setdefault(path.stem, path)
    label_path = next(root.rglob("labels.csv"))
    events_of_audio = {}
    with open(label_path, encoding="utf-8", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["calltype"])) in MARMOSET_NOT_VOCAL:
                continue
            onset = float(row["start"])
            offset = float(row["end"])
            if offset <= onset:
                continue
            name = Path(row["filename"]).name
            audio = audio_of_name.get(name) or audio_of_name.get(Path(name).stem)
            if audio is None:
                continue
            event = (onset, offset, TAXON_OF_CORPUS["infant_marmosets"])
            events_of_audio.setdefault(audio, []).append(event)
    recordings = []
    for audio, events in events_of_audio.items():
        recordings.append(recording("infant_marmosets", audio, audio.stem, events))
    return recordings


# ---------------------------------------------------------------- zebra finch pairs
def read_tworadiobirds():
    """Zebra finch pairs: tables <date>_male_voc.csv and <date>_female_voc.csv (file, onset
    and duration in samples)."""
    root = corpus_dir("tworadiobirds")
    audio_of_name = {}
    for path in root.rglob("*.w64"):
        audio_of_name[path.name] = path
    samples_of_audio = {}
    # The table of the male is read before that of the female.
    for table_path in sorted(root.rglob("*_voc.csv"), reverse=True):
        with open(table_path) as handle:
            for row in csv.DictReader(handle):
                audio = audio_of_name.get(row["file"].strip())
                if audio is None:
                    continue
                onset = float(row["onset"])
                duration = float(row["duration"])
                samples_of_audio.setdefault(audio, []).append((onset, onset + duration))
    recordings = []
    for audio, sample_pairs in samples_of_audio.items():
        rate = soundfile.info(str(audio)).samplerate
        events = []
        for onset, offset in sample_pairs:
            events.append((onset / rate, offset / rate, TAXON_OF_CORPUS["tworadiobirds"]))
        recordings.append(recording("tworadiobirds", audio, audio.stem, events))
    return recordings


# ---------------------------------------------------------------- Wytham great tits
def read_wytham():
    """Wytham great tits: crowsetta.csv, one row per note (notated_path, onset_s, offset_s)."""
    root = corpus_dir("wytham")
    audio_of_name = {}
    for path in root.rglob("*.wav"):
        audio_of_name[path.name] = path
    events_of_audio = {}
    with open(root / "crowsetta.csv") as handle:
        for row in csv.DictReader(handle):
            audio = audio_of_name.get(Path(row["notated_path"]).name)
            if audio is None:
                continue
            onset = float(row["onset_s"])
            offset = float(row["offset_s"])
            if offset > onset:
                event = (onset, offset, TAXON_OF_CORPUS["wytham"])
                events_of_audio.setdefault(audio, []).append(event)
    recordings = []
    for audio, events in events_of_audio.items():
        recordings.append(recording("wytham", audio, audio.stem, events))
    return recordings


# ---------------------------------------------------------------- Sierra Nevada soundscapes
def read_sne_full():
    """Sierra Nevada soundscapes: annotations.csv with eBird codes, species.csv with their
    scientific names. The code ???? (unidentified) has the name "unknown"."""
    root = corpus_dir("sne_full")
    name_of_code = {}
    with open(root / "species.csv") as handle:
        for row in csv.DictReader(handle):
            name_of_code[row["Species eBird Code"].strip()] = row["Scientific Name"].strip()
    audio_of_name = {}
    for path in root.rglob("*.flac"):
        audio_of_name.setdefault(path.name, path)
    events_of_audio = {}
    with open(root / "annotations.csv") as handle:
        for row in csv.DictReader(handle):
            audio = audio_of_name.get(row["Filename"].strip())
            if audio is None:
                continue
            onset = float(row["Start Time (s)"])
            offset = float(row["End Time (s)"])
            taxon = name_of_code.get(row["Species eBird Code"].strip())
            if not taxon or offset <= onset:
                continue
            events_of_audio.setdefault(audio, []).append((onset, offset, taxon))
    recordings = []
    for audio, events in events_of_audio.items():
        recordings.append(recording("sne_full", audio, audio.stem, events))
    return recordings


READERS = {
    "budgie": read_budgie,
    "canary": read_canary,
    "enabirds": read_enabirds,
    "infant_marmosets": read_infant_marmosets,
    "nbm": read_nbm,
    "ozf": read_ozf,
    "right_whale": read_right_whale,
    "rodent_usv": read_rodent_usv,
    "rodents2019": read_rodents2019,
    "singing_mice": read_singing_mice,
    "sne_full": read_sne_full,
    "spiders": read_spiders,
    "tworadiobirds": read_tworadiobirds,
    "wytham": read_wytham,
}


def recording_order(item):
    """Sort key of a recording: corpus, identifier, audio path."""
    return (item["corpus"], item["rec_id"], item["audio"])


def read_all():
    """Return the recordings of all corpora in a fixed order."""
    recordings = []
    for corpus in sorted(READERS):
        recordings.extend(READERS[corpus]())
    return sorted(recordings, key=recording_order)


def main():
    """Print the number of recordings, vocalizations and taxa of every corpus."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)
    print(f"{'corpus':18s} {'recordings':>10s} {'vocalizations':>14s} {'taxa':>5s}")
    for corpus in sorted(READERS):
        recordings = READERS[corpus]()
        events = 0
        taxa = set()
        for item in recordings:
            events += len(item["events"])
            for _, _, taxon in item["events"]:
                if taxon is not None:
                    taxa.add(taxon)
        print(f"{corpus:18s} {len(recordings):10d} {events:14d} {len(taxa):5d}")


if __name__ == "__main__":
    main()
