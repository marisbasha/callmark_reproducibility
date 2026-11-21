"""Write the identification predictions of the Voxaboxen identification model.

    python -m systems.voxaboxen.predict_identification --data DIR --size 50

Reads the test predictions that the training run wrote to
models/voxaboxen/identification_<size>/run/outputs/. The predicted taxon of an example is the
label of the predicted box with the largest intersection over union with the vocalization in
its clip; an example without an overlapping box is predicted as "__none__". Writes
predictions/voxaboxen/identification_<size>.json.
"""
import argparse

from benchmark import config, data
from systems.voxaboxen import files

NO_PREDICTION = "__none__"


def best_overlapping_label(predictions, onset, offset):
    """Return the label of the box that overlaps (onset, offset) most, the first of equals."""
    best_label = None
    best_overlap = 0.0
    for box_onset, box_offset, label in predictions:
        intersection = max(0.0, min(box_offset, offset) - max(box_onset, onset))
        union = max(box_offset, offset) - min(box_onset, onset)
        overlap = 0.0
        if union > 0:
            overlap = intersection / union
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    return best_label


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="", help="data root (default: CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    output_dir = files.run_dir("identification", arguments.size) / "outputs"
    input_dir = files.input_dir("identification", arguments.size)
    predicted = {}
    for example in data.read_json(input_dir / "test_examples.json"):
        boxes = files.read_predictions(output_dir, example["name"], 1)
        label = best_overlapping_label(boxes, example["onset"], example["offset"])
        predicted[example["example_id"]] = label or NO_PREDICTION
    target = config.prediction_file("voxaboxen", "identification", arguments.size)
    data.write_json(target, predicted)
    print(f"{len(predicted)} examples -> {target}")


if __name__ == "__main__":
    main()
