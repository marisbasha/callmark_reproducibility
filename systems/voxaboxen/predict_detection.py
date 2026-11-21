"""Write the detection predictions of the Voxaboxen detection model.

    python -m systems.voxaboxen.predict_detection --data DIR --size 50

Reads the test predictions that the training run wrote to
models/voxaboxen/detection_<size>/run/outputs/ and writes
predictions/voxaboxen/detection_<size>.json.gz with the times of slowed chunks divided by their
slow-down factor.
"""
import argparse

from benchmark import config, data
from systems.voxaboxen import files


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="", help="data root (default: CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    output_dir = files.run_dir("detection", arguments.size) / "outputs"
    input_dir = files.input_dir("detection", arguments.size)
    slow_down = data.read_json(input_dir / "test_slow_down.json")
    rows = []
    for chunk_id in data.dataset_reference():
        predicted = []
        for onset, offset, _ in files.read_predictions(output_dir, chunk_id, slow_down[chunk_id]):
            predicted.append((onset, offset))
        rows.append({"chunk_id": chunk_id, "pred": predicted})
    target = config.prediction_file("voxaboxen", "detection", arguments.size)
    data.write_json(target, rows)
    print(f"{len(rows)} chunks -> {target}")


if __name__ == "__main__":
    main()
