"""Detection+Identification with DualWhisper: one decoding pass per chunk.

    python -m systems.dualwhisper.predict_end_to_end --data DIR --size 50

Decodes every chunk of labels/end_to_end_chunks.json with the model of
models/dualwhisper/<size>/final_checkpoint_ct2 and writes
predictions/dualwhisper/end_to_end_<size>.json.gz. The model emits onset, taxon and offset of
each vocalization; the acoustic settings of each chunk are those of the prepared test set.
"""
import argparse
import json
import sys

import soundfile as sf

from benchmark import config, data
from systems.whisperseg import common as whisperseg

sys.path.insert(0, str(whisperseg.WHISPERSEG))
from model import WhisperSegmenterFast  # noqa: E402

BATCH_SIZE = 8
# Number of left-padded views of a chunk that are decoded and combined.
NUM_VIEWS = 3


def predict_chunk(segmenter, chunk_id):
    """Return [[onset, offset, taxon], ...] of one test chunk, in seconds from its start."""
    test_dir = config.prepared_dir("test")
    settings = json.loads((test_dir / f"{chunk_id}.json").read_text())
    waveform, sampling_rate = sf.read(str(test_dir / f"{chunk_id}.wav"), dtype="float32")
    prediction = segmenter.segment(
        waveform,
        sampling_rate,
        min_frequency=settings["min_frequency"],
        spec_time_step=settings["spec_time_step"],
        min_segment_length=settings["min_segment_length"],
        eps=settings["eps"],
        time_per_frame_for_voting=settings["time_per_frame_for_scoring"],
        batch_size=BATCH_SIZE,
        num_trials=NUM_VIEWS,
    )
    vocalizations = []
    for onset, offset, taxon in zip(
        prediction["onset"], prediction["offset"], prediction["cluster"]
    ):
        vocalizations.append([onset, offset, taxon])
    return vocalizations


def predict(chunk_ids, size):
    """Return the prediction rows of test chunks `chunk_ids` with the model of one size."""
    model_dir = config.DATA / "models" / "dualwhisper" / str(size)
    segmenter = WhisperSegmenterFast(str(model_dir / "final_checkpoint_ct2"), device="cuda")
    rows = []
    for index, chunk_id in enumerate(chunk_ids):
        rows.append({"chunk_id": chunk_id, "pred": predict_chunk(segmenter, chunk_id)})
        if index % 100 == 0:
            print(f"{index}/{len(chunk_ids)} chunks", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: environment variable CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    chunk_ids = data.read_json(config.label_file("end_to_end_chunks.json"))
    rows = predict(chunk_ids, arguments.size)
    output = config.prediction_file("dualwhisper", "end_to_end", arguments.size)
    data.write_json(output, rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
