"""Download the public original corpora into corpora/<name>/ of the data root.

    python -m systems.sources.download_corpora --data DIR [--corpora NAME ...]

Archives are kept in corpora/_archives/ and are not downloaded again when they exist. The
budgerigar and singing-mouse recordings are not in a public archive (README.md).
"""
import argparse
import shutil
import tarfile
import zipfile

import requests

from benchmark import config
from systems.sources import corpora

ZENODO = "https://zenodo.org/records/{record}/files/{name}?download=1"
OSF = "https://osf.io/download/{identifier}/"

# Corpus -> files of its Zenodo record.
ZENODO_FILES = {
    "canary": (6521932, ["M1-2016-sping_audio.zip",
                         "M1-2016-spring_audacity_annotations.zip"]),
    "enabirds": (4656848, ["wav_Files.zip", "annotation_Files.zip"]),
    "infant_marmosets": (10130104, ["InfantMarmosetsVox_documentation.tar.gz",
                                    "InfantMarmosetsVox_twin_1.tar.gz",
                                    "InfantMarmosetsVox_twin_2.tar.gz",
                                    "InfantMarmosetsVox_twin_3.tar.gz",
                                    "InfantMarmosetsVox_twin_4.tar.gz",
                                    "InfantMarmosetsVox_twin_5.tar.gz"]),
    "nbm": (17573913, ["zenodo_nbm_db.zip"]),
    "ozf": (15507508, ["OZF.zip"]),
    "right_whale": (15843741, ["RW.zip"]),
    "rodent_usv": (3428024, ["gerbil.zip", "mouse_B6pup.zip", "mouse_BALBc.zip",
                             "mouse_C57BL.zip", "mouse_Shank2.zip", "rat_distressed.zip",
                             "rat_pleasant.zip"]),
    "sne_full": (7525805, ["annotations.csv", "species.csv", "soundscape_data.zip"]),
    "spiders": (15843741, ["JS.zip"]),
    "tworadiobirds": (7253729, ["TwoRadiobirds.zip"]),
}

# Files of the Wytham great tit record of the Open Science Framework (osf.io/n8ac9).
WYTHAM_ANNOTATIONS = "4cmjv"                                # great-tit-hits-crowsetta.csv
WYTHAM_AUDIO_PARTS = ["xywcm", "5ytvd", "cez45", "6x243"]   # song-files.zip.part1 to part4


def download(url, path):
    """Download `url` to `path` unless the file exists."""
    if path.exists():
        return
    print(f"downloading {path.name}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(partial, "wb") as handle:
            for block in response.iter_content(chunk_size=1 << 20):
                handle.write(block)
    partial.rename(path)


def unpack(archive, target, name):
    """Extract a zip or tar.gz archive into `target`; any other file is copied there as
    `name`."""
    target.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as opened:
            opened.extractall(target)
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive) as opened:
            opened.extractall(target)
    else:
        shutil.copy(archive, target / name)


def move_children_up(folder):
    """Move the content of `folder` into its parent folder and remove `folder`."""
    for child in list(folder.iterdir()):
        child.rename(folder.parent / child.name)
    folder.rmdir()


def download_zenodo_corpus(corpus, archives):
    """Download and unpack the files of one corpus and arrange them as its reader expects."""
    record, names = ZENODO_FILES[corpus]
    target = corpora.corpus_dir(corpus)
    for name in names:
        archive = archives / f"{corpus}__{name}"
        download(ZENODO.format(record=record, name=name), archive)
        unpack(archive, target, name)
    if corpus == "enabirds":
        # The reader looks for <name>.wav next to each selection table.
        for audio in target.glob("*/*.WAV"):
            audio.rename(audio.with_suffix(".wav"))
    if corpus == "nbm":
        move_children_up(target / "zenodo_nbm_db")
    if corpus == "spiders":
        move_children_up(target / "JS")
    if corpus == "right_whale":
        move_children_up(target / "RW")


def download_wytham(archives):
    """Download the annotations and the four audio archives of the great tits.

    Every part of song-files.zip is a complete zip archive with the folders WAV and JSON."""
    target = corpora.corpus_dir("wytham")
    target.mkdir(parents=True, exist_ok=True)
    download(OSF.format(identifier=WYTHAM_ANNOTATIONS), target / "crowsetta.csv")
    for number, identifier in enumerate(WYTHAM_AUDIO_PARTS, start=1):
        archive = archives / f"wytham__song-files.part{number}.zip"
        download(OSF.format(identifier=identifier), archive)
        unpack(archive, target, archive.name)


def main():
    """Download the requested corpora."""
    public = sorted(list(ZENODO_FILES) + ["wytham"])
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    parser.add_argument("--corpora", nargs="+", choices=public, default=public)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    archives = config.DATA / "corpora" / "_archives"
    archives.mkdir(parents=True, exist_ok=True)
    for corpus in arguments.corpora:
        if corpus == "wytham":
            download_wytham(archives)
        else:
            download_zenodo_corpus(corpus, archives)
        print(f"{corpus}: done", flush=True)


if __name__ == "__main__":
    main()
