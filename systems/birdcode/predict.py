"""Write the prediction files of the fine-tuned BirdCODE models.

    python -m systems.birdcode.predict --data DIR --task detection --size 500
    python -m systems.birdcode.predict --data DIR --task end_to_end --size 500
    python -m systems.birdcode.predict --data DIR --task identification --size 500

detection and end_to_end use models/birdcode/detector_<size>, identification uses
models/birdcode/identifier_<size>. Output: predictions/birdcode/<task>_<size>.json[.gz].
"""
import argparse
import csv
import io

import numpy
import pyarrow.parquet as pq
import soundfile
import torch
from esp_research.protocols.detector import DetectorOutput
from sound_event_detection.inference.lsi_postprocess_cli import (
    postprocess_and_convert_detector_output_to_selection_table as selection_table,
)

from benchmark import config, data
from systems.birdcode import birdcode

# The input is run this many times, each time starting a further 1/8 of an output frame later.
SHIFTS = 8
WINDOW_OVERLAP = 0.5
WINDOW_BATCH_SIZE = 4
PROBABILITY_THRESHOLD = 0.5
# Of two predicted vocalizations with an intersection over union above 0.8, across taxa,
# only the one with the higher score is kept.
POST_PROCESSING = {"nms": {"iou_threshold": 0.8}}


def frame_probabilities(model, waveform):
    """Return the taxon probabilities of every output frame of a waveform."""
    output = model.run(waveform[None, :], overlap=WINDOW_OVERLAP, batch_size=WINDOW_BATCH_SIZE)
    return output.predictions[0]


def shifted_probabilities(model, waveform):
    """Return frame probabilities on a grid SHIFTS times finer than the output frames.

    The model is run on the waveform with its first 0, 1/8, ... 7/8 of a frame removed, and
    the frames of the runs are interleaved in time order."""
    samples_per_frame = model.sample_rate / model.frame_rate
    runs = []
    for shift in range(SHIFTS):
        first_sample = int(round(shift * samples_per_frame / SHIFTS))
        runs.append(frame_probabilities(model, waveform[first_sample:]))
    frame_count = min(len(run) for run in runs)
    aligned = numpy.stack([run[:frame_count] for run in runs], 1)
    interleaved = aligned.reshape(frame_count * SHIFTS, -1)
    return DetectorOutput(
        predictions=interleaved[None],
        frame_rate=model.frame_rate * SHIFTS,
        class_names=list(model.labels),
    )


def predicted_vocalizations(model, waveform):
    """Return [onset, offset, taxon] of every vocalization predicted in a waveform.

    Runs of frames with a probability of at least 0.5 for a taxon become vocalizations of
    that taxon; non-maximum suppression across taxa follows."""
    probabilities = shifted_probabilities(model, waveform)
    table = selection_table(probabilities, PROBABILITY_THRESHOLD, POST_PROCESSING)
    vocalizations = []
    for row in csv.DictReader(io.StringIO(table), delimiter="\t"):
        onset = float(row["Begin Time (s)"])
        offset = float(row["End Time (s)"])
        vocalizations.append([onset, offset, row["Species"]])
    return vocalizations


def detection(model):
    """Return the detection predictions of every prepared test chunk.

    Chunks sampled above 64 kHz get no predictions."""
    rows = []
    for file in sorted(config.prepared_dir("test").glob("*.json")):
        annotation = birdcode.read_json(file)
        boundaries = []
        if annotation["sr"] <= birdcode.HIGHEST_SAMPLING_RATE:
            waveform = birdcode.load_audio(file.with_suffix(".wav"), model)
            for onset, offset, _ in predicted_vocalizations(model, waveform):
                boundaries.append([onset, offset])
        rows.append({"chunk_id": file.stem, "pred": boundaries})
    return rows


def end_to_end(model):
    """Return the predictions with taxa for the chunks of the Detection+Identification subset."""
    rows = []
    for chunk_id in birdcode.read_json(config.label_file("end_to_end_chunks.json")):
        path = config.prepared_dir("test") / f"{chunk_id}.wav"
        waveform = birdcode.load_audio(path, model)
        rows.append({"chunk_id": chunk_id, "pred": predicted_vocalizations(model, waveform)})
    return rows


def identification_examples():
    """Yield the identification test examples with their audio, in dataset order."""
    columns = ["audio", "dataset", "chunk_id", "event_index", "onset_in_window",
               "offset_in_window"]
    files = sorted((config.dataset_dir() / "classification").glob("test-*.parquet"))
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                if row["dataset"] != config.DUPLICATE_CORPUS:
                    yield row


def identification(model):
    """Return {example_id: taxon} for every identification test example.

    The vocalization is centred in a 5-s clip of silence as in training. The taxon with the
    highest mean probability over the output frames that the vocalization touches is chosen."""
    frame_rate = model.frame_rate
    labels = list(model.labels)
    predicted = {}
    for example in identification_examples():
        audio = io.BytesIO(example["audio"]["bytes"])
        waveform, sampling_rate = soundfile.read(audio, dtype="float32")
        clip, onset, offset = birdcode.centre_vocalization(
            waveform,
            sampling_rate,
            float(example["onset_in_window"]),
            float(example["offset_in_window"]),
        )
        clip = birdcode.to_model_rate(clip, sampling_rate, model)
        probabilities = frame_probabilities(model, clip)
        first_frame = int(numpy.floor(onset * frame_rate))
        end_frame = max(int(numpy.ceil(offset * frame_rate)), first_frame + 1)
        mean_probabilities = probabilities[first_frame:end_frame].mean(0)
        predicted[data.example_id(example)] = labels[int(mean_probabilities.argmax())]
    return predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data root (default: environment variable CALLMARK_DATA)")
    parser.add_argument("--task", required=True,
                        choices=("detection", "end_to_end", "identification"))
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    args = parser.parse_args()
    config.use_data_root(args.data)

    if args.task == "identification":
        model_name = "identifier"
    else:
        model_name = "detector"
    model = birdcode.load_finetuned(birdcode.model_dir(model_name, args.size))
    with torch.no_grad():
        if args.task == "detection":
            predictions = detection(model)
        elif args.task == "end_to_end":
            predictions = end_to_end(model)
        else:
            predictions = identification(model)
    data.write_json(config.prediction_file("birdcode", args.task, args.size), predictions)
    print("wrote", config.prediction_file("birdcode", args.task, args.size))


if __name__ == "__main__":
    main()
