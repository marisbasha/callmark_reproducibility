"""Pair xeno-canto vocalizations with background stretches of benchmark training audio.

    python -m systems.whisperid.pretraining.build_bridge_manifest --data DIR \
        --vocalizations FILE --cap 1000 --out FILE

Background stretches are the parts of whisperid/background_chunks without annotated
vocalization that last at least 0.75 s. Every vocalization of the input manifest (at most
--cap per taxon) gets a random stretch; half of them also get a second vocalization, which
is kept when its taxon differs.
"""
import argparse
import json
import random
from collections import defaultdict

import soundfile

from benchmark import config

SEED = 42
MIN_STRETCH_SECONDS = 0.75
SECOND_VOCALIZATION_SHARE = 0.5
# File name prefixes of the corpora that supply backgrounds.
BACKGROUND_PREFIXES = ("enabirds", "sne", "nbm", "wytham")


def unannotated_stretches(onsets, offsets, duration):
    """Return the (start, end) stretches of a chunk without annotated vocalization."""
    stretches = []
    position = 0.0
    for onset, offset in sorted(zip(onsets, offsets)):
        if onset - position >= MIN_STRETCH_SECONDS:
            stretches.append((position, onset))
        position = max(position, offset)
    if duration - position >= MIN_STRETCH_SECONDS:
        stretches.append((position, duration))
    return stretches


def background_stretches(directory):
    """Return the background stretches of all chunks of the background corpora."""
    backgrounds = []
    for label_file in sorted(directory.glob("*.json")):
        if not label_file.stem.startswith(BACKGROUND_PREFIXES):
            continue
        audio_file = label_file.with_suffix(".wav")
        with open(label_file) as handle:
            labels = json.load(handle)
        information = soundfile.info(str(audio_file))
        duration = information.frames / information.samplerate
        onsets = labels.get("onset", [])
        offsets = labels.get("offset", [])
        for start, end in unannotated_stretches(onsets, offsets, duration):
            backgrounds.append({
                "audio": str(audio_file),
                "start": round(start, 3),
                "end": round(end, 3),
                "dataset": label_file.stem.split("__")[0],
            })
    return backgrounds


def capped_vocalizations(manifest, cap, generator):
    """Return the vocalizations of a manifest, at most `cap` per taxon, in random order."""
    by_taxon = defaultdict(list)
    with open(manifest) as handle:
        for line in handle:
            item = json.loads(line)
            by_taxon[item["species"]].append(item)
    vocalizations = []
    for taxon in sorted(by_taxon):
        items = by_taxon[taxon]
        if len(items) > cap:
            items = generator.sample(items, cap)
        vocalizations.extend(items)
    generator.shuffle(vocalizations)
    return vocalizations


def main():
    """Write the manifest of mixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--vocalizations", required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config.use_data_root(args.data)

    backgrounds = background_stretches(config.DATA / "whisperid" / "background_chunks")
    generator = random.Random(SEED)
    vocalizations = capped_vocalizations(args.vocalizations, args.cap, generator)
    with open(args.out, "w") as handle:
        for vocalization in vocalizations:
            background = backgrounds[generator.randrange(len(backgrounds))]
            item = {
                "audio": vocalization["audio"],
                "species": vocalization["species"],
                "onset": vocalization["onset"],
                "offset": vocalization["offset"],
                "bg_audio": background["audio"],
                "bg_start": background["start"],
                "bg_end": background["end"],
                "bg_dataset": background["dataset"],
            }
            if generator.random() < SECOND_VOCALIZATION_SHARE:
                second = vocalizations[generator.randrange(len(vocalizations))]
                if second["species"] != vocalization["species"]:
                    item["overlay_audio"] = second["audio"]
                    item["overlay_onset"] = second["onset"]
                    item["overlay_offset"] = second["offset"]
            handle.write(json.dumps(item) + "\n")
    print(f"{len(vocalizations)} mixtures, {len(backgrounds)} background stretches", flush=True)


if __name__ == "__main__":
    main()
