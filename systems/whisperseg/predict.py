"""Detect the vocalizations of every test chunk with a fine-tuned WhisperSeg model.

    python -m systems.whisperseg.predict --system whisperseg --size 50 [--data DIR]

Reads prepared/test and models/<system>/<size>/final_checkpoint_ct2, writes
predictions/<system>/detection_<size>.json.gz.
"""
import argparse
import json
import sys

import soundfile

from benchmark import config, data
from systems.whisperseg import common

sys.path.insert(0, str(common.WHISPERSEG))
from model import WhisperSegmenterFast  # noqa: E402

# Number of left-padding views of a chunk that are decoded and combined by voting.
NUM_VIEWS = 3
BATCH_SIZE = 8


def detect(segmenter, wav_path):
    """Return the [onset, offset] pairs predicted for one chunk, in seconds from its start.

    The spectrogram and voting settings are those stored next to the audio."""
    settings = json.loads(wav_path.with_suffix(".json").read_text())
    audio, sampling_rate = soundfile.read(wav_path, dtype="float32")
    prediction = segmenter.segment(
        audio,
        sampling_rate,
        min_frequency=settings["min_frequency"],
        spec_time_step=settings["spec_time_step"],
        min_segment_length=settings["min_segment_length"],
        eps=settings["eps"],
        batch_size=BATCH_SIZE,
        num_trials=NUM_VIEWS,
    )
    pairs = []
    for onset, offset in zip(prediction["onset"], prediction["offset"]):
        pairs.append([onset, offset])
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--system", required=True, choices=common.SYSTEMS)
    parser.add_argument("--size", required=True, type=int, choices=config.SIZES)
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    checkpoint = common.model_dir(arguments.system, arguments.size) / "final_checkpoint_ct2"
    segmenter = WhisperSegmenterFast(str(checkpoint), device="cuda")
    wav_paths = sorted(config.prepared_dir("test").glob("*.wav"))
    rows = []
    for index, wav_path in enumerate(wav_paths):
        rows.append({"chunk_id": wav_path.stem, "pred": detect(segmenter, wav_path)})
        if (index + 1) % 100 == 0:
            print(f"{index + 1} of {len(wav_paths)} chunks", flush=True)
    output = config.prediction_file(arguments.system, "detection", arguments.size)
    data.write_json(output, rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
