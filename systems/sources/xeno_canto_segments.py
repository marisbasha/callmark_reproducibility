"""Detect the vocalizations of the selected xeno-canto recordings with WhisperSeg.

    python -m systems.sources.xeno_canto_segments --data DIR --model CHECKPOINT

Reads xeno_canto/selected_recordings.jsonl and xeno_canto/recordings/ and writes
xeno_canto/segments.jsonl: the same lines with "segments": {"onset": [...], "offset": [...],
"cluster": [...]} (seconds) added to every recording. A recording that cannot be read or
segmented gets "segments": null.

CHECKPOINT is a WhisperSeg checkpoint in CTranslate2 format. WhisperSeg is imported from an
unmodified clone of https://github.com/nianlonggu/WhisperSeg in systems/sources/WhisperSeg.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa

from benchmark import config

sys.path.insert(0, str(Path(__file__).resolve().parent / "WhisperSeg"))


def main():
    """Segment every selected recording at its own sampling rate."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    parser.add_argument("--model", required=True, help="WhisperSeg checkpoint (CTranslate2)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    from model import WhisperSegmenterFast
    segmenter = WhisperSegmenterFast(arguments.model, "cuda")

    directory = config.DATA / "xeno_canto"
    with open(directory / "selected_recordings.jsonl") as source, \
            open(directory / "segments.jsonl", "w") as output:
        for line in source:
            entry = json.loads(line)
            for recording in entry["data"]:
                path = directory / "recordings" / recording["file_name"]
                try:
                    audio, rate = librosa.load(path, sr=None, mono=True)
                    recording["segments"] = segmenter.segment(audio, sr=rate)
                except Exception:
                    print(f"not segmented: {recording['file_name']}", flush=True)
                    recording["segments"] = None
            output.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
