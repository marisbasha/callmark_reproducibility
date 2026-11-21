"""Write the Detection+Identification predictions of the Voxaboxen taxon model.

    python -m systems.voxaboxen.predict_end_to_end --data DIR --voxaboxen CHECKOUT --size 50

Runs the taxon model of the training size over the chunks of labels/end_to_end_chunks.json.
Forward and backward predictions are combined at the threshold that the detection model of the
same training size chose on its validation files (read from its train.log), not at the
threshold of the taxon model. Writes

    models/voxaboxen/taxon_<size>/run/inference/       Voxaboxen selection tables
    predictions/voxaboxen/end_to_end_<size>.json.gz
"""
import argparse
import csv
import os
import re
import subprocess
import sys

from benchmark import config, data
from systems.voxaboxen import files


def detection_model_threshold(size):
    """Return the combination threshold printed by the training of the detection model."""
    log = (files.project_dir("detection", size) / "train.log").read_text()
    match = re.search(r"Found best thresh on val set: .*comb_discard=([0-9.]+),", log)
    return match.group(1)


def write_file_list(size, chunk_ids, target):
    """Write the Voxaboxen file list of the given test chunks."""
    wanted = set(chunk_ids)
    rows = []
    with open(files.input_dir("taxon", size) / "test_info.csv") as handle:
        for row in csv.DictReader(handle):
            if row["fn"] in wanted:
                rows.append((row["fn"], row["audio_fp"], row["selection_table_fp"]))
    files.write_info(target, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="", help="data root (default: CALLMARK_DATA)")
    parser.add_argument("--voxaboxen", required=True, help="Voxaboxen checkout with the changes")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)
    size = arguments.size

    run_dir = files.run_dir("taxon", size).resolve()
    chunk_ids = data.read_json(config.label_file("end_to_end_chunks.json"))
    file_list = run_dir / "end_to_end_info.csv"
    write_file_list(size, chunk_ids, file_list)
    command = [
        sys.executable, "main.py", "inference",
        f"--model-args-fp={run_dir / 'params.yaml'}",
        f"--model-checkpoint-fp={run_dir / 'final-model.pt'}",
        f"--file-info-for-inference={file_list}",
        "--detection-threshold=0.5",
        "--classification-threshold=0.0",
        "--comb-iou-threshold=0.5",
        f"--comb-discard-threshold={detection_model_threshold(size)}",
    ]
    environment = dict(os.environ, VOXABOXEN_EVAL_WORKERS="0")
    with open(run_dir / "inference.log", "w") as log:
        subprocess.run(command, cwd=arguments.voxaboxen, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)

    slow_down = data.read_json(files.input_dir("taxon", size) / "test_slow_down.json")
    rows = []
    for chunk_id in chunk_ids:
        predicted = files.read_predictions(run_dir / "inference", chunk_id, slow_down[chunk_id])
        rows.append({"chunk_id": chunk_id, "pred": predicted})
    target = config.prediction_file("voxaboxen", "end_to_end", size)
    data.write_json(target, rows)
    print(f"{len(rows)} chunks -> {target}")


if __name__ == "__main__":
    main()
