"""Train one Voxaboxen model and predict the test set with it.

    python -m systems.voxaboxen.train --data DIR --voxaboxen CHECKOUT --model detection --size 50

`--model` is detection, taxon or identification. The Voxaboxen checkout sets up a project in
models/voxaboxen/<model>_<size>/ and trains the run models/voxaboxen/<model>_<size>/run/:

    run/final-model.pt     weights after the last epoch
    run/params.yaml        all settings
    run/outputs/           predictions for the validation and test files, combined from the
                           forward and backward passes at the threshold chosen on validation
    train.log              output of the training, with the chosen threshold
"""
import argparse
import os
import subprocess
import sys

from benchmark import config
from systems.voxaboxen import files

# 50 epochs without early stopping; batches of 16 clips with gradients accumulated over two
# batches; BirdAVES encoder, frozen for the first three epochs (Voxaboxen default); seed 0.
TRAINING_ARGUMENTS = [
    "--n-epochs=50",
    "--batch-size=16",
    "--grad-accum=2",
    "--lr=0.00005",
    "--bidirectional",
    "--encoder-type=aves",
    "--num-workers=4",
]
# The identification clips are 3 s long: clips of 1 s every 0.5 s instead of 10 s every 5 s.
IDENTIFICATION_ARGUMENTS = ["--clip-duration=1.0", "--clip-hop=0.5"]


def run_voxaboxen(checkout, mode, mode_arguments, log):
    """Run one mode of main.py of the Voxaboxen checkout, appending its output to the log."""
    command = [sys.executable, "main.py", mode] + mode_arguments
    # The per-file loaders of the evaluation load audio in the main process.
    environment = dict(os.environ, VOXABOXEN_EVAL_WORKERS="0")
    subprocess.run(command, cwd=checkout, env=environment, stdout=log,
                   stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="", help="data root (default: CALLMARK_DATA)")
    parser.add_argument("--voxaboxen", required=True, help="Voxaboxen checkout with the changes")
    parser.add_argument("--model", required=True, choices=files.MODELS)
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    parser.add_argument("--select-on-validation", action="store_true",
                        help="validate after every epoch and keep the best weights (Voxaboxen's "
                             "early stopping), instead of the weights of the last epoch")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    input_dir = files.input_dir(arguments.model, arguments.size).resolve()
    project_dir = files.project_dir(arguments.model, arguments.size).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    training_arguments = [
        f"--project-config-fp={project_dir / 'project_config.yaml'}",
        f"--name={files.RUN_NAME}",
    ]
    training_arguments += TRAINING_ARGUMENTS
    if arguments.model == "identification":
        training_arguments += IDENTIFICATION_ARGUMENTS
    if arguments.select_on_validation:
        training_arguments += ["--val-during-training", "--early-stopping"]
    with open(project_dir / "train.log", "w") as log:
        setup_arguments = [f"--data-dir={input_dir}", f"--project-dir={project_dir}"]
        run_voxaboxen(arguments.voxaboxen, "project-setup", setup_arguments, log)
        run_voxaboxen(arguments.voxaboxen, "train-model", training_arguments, log)
    print(f"trained {files.run_dir(arguments.model, arguments.size)}")


if __name__ == "__main__":
    main()
