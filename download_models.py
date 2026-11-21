"""Download the trained models of the paper from Hugging Face into the data root.

    python download_models.py SYSTEM [SYSTEM ...] [--data DIR]
    python download_models.py all

The models are placed where the prediction scripts read them:

    $CALLMARK_DATA/models/<system>/...                               benchmark systems
    $CALLMARK_DATA/multichannel/models/<system>/ct2                  MC-WhisperSeg and the
                                                                     two single-sensor models
    $CALLMARK_DATA/added_audio/<set>/multichannel/models/microphone/ct2
                                                                     microphone-only detector
                                                                     trained with added audio

Requires huggingface_hub.
"""
import argparse
import os
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

OWNER = "songbirdini"
SIZES_GB = {
    "whisperseg": 51,
    "whisperid": 46,
    "dualwhisper": 22,
    "voxaboxen": 6,
    "birdcode": 2,
    "naturelm": 47,
    "biolingual": 2,
    "tweetynet": 0.01,
    "das": 0.01,
    "mc-whisperseg": 18,
}
PLACEHOLDER = "DATA_ROOT"
CARD = ["README.md", ".gitattributes"]


def repository(system):
    """Name of the Hugging Face repository with the models of a system."""
    return f"{OWNER}/callmark-{system}"


def link_into(source, target):
    """Link every file below the folder source to the same relative path below target."""
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        link = target / path.relative_to(source)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path.resolve())


def place_multichannel_models(download, root):
    """Link the models of the multi-channel part to the folders its scripts read."""
    for folder in sorted(download.iterdir()):
        if folder.name == "pairs":
            link_into(folder, root / "multichannel" / "models")
        elif folder.name.startswith("added_audio_"):
            added_set = folder.name[len("added_audio_"):]
            target = root / "added_audio" / added_set / "multichannel" / "models"
            link_into(folder, target)


def finish_voxaboxen(root):
    """Unpack the test outputs of every Voxaboxen run and write the data root into the
    settings, in which Voxaboxen stores absolute paths."""
    folder = root / "models" / "voxaboxen"
    for archive in sorted(folder.rglob("outputs.tar")):
        if not (archive.parent / "outputs").exists():
            with tarfile.open(archive) as file:
                file.extractall(archive.parent)
    for settings in sorted(folder.rglob("params.yaml")):
        text = settings.read_text()
        if PLACEHOLDER in text:
            settings.write_text(text.replace(PLACEHOLDER, str(root)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("systems", nargs="+", choices=sorted(SIZES_GB) + ["all"])
    parser.add_argument("--data", help="data root (default: $CALLMARK_DATA)")
    args = parser.parse_args()
    root = args.data or os.environ.get("CALLMARK_DATA")
    if not root:
        raise SystemExit("Set CALLMARK_DATA or pass --data")
    root = Path(root).resolve()

    systems = args.systems
    if "all" in systems:
        systems = sorted(SIZES_GB)
    for system in systems:
        print(f"{system}: {repository(system)}, {SIZES_GB[system]} GB", flush=True)
        if system == "mc-whisperseg":
            download = root / "downloads" / "mc-whisperseg"
            snapshot_download(repository(system), local_dir=download, ignore_patterns=CARD)
            place_multichannel_models(download, root)
        else:
            snapshot_download(
                repository(system), local_dir=root / "models", ignore_patterns=CARD)
        if system == "voxaboxen":
            finish_voxaboxen(root)


if __name__ == "__main__":
    main()
