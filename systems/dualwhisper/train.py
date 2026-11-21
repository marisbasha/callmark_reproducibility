"""Train DualWhisper at one training size.

    python -m systems.dualwhisper.train --data DIR --size 50

Runs the WhisperSeg training script (through whisperseg_training.py, which sets the decoding
used in validation) on prepared/train_<size>_joined/, in which the label of each vocalization
is its taxon, starting from the WhisperSeg detector of the same size
(models/whisperseg/<size>/final_checkpoint). The model is validated after every
epoch; training stops after 12 epochs or when the best validation score is four epochs old.
The checkpoint with the highest validation score is written to
models/dualwhisper/<size>/final_checkpoint and, converted to CTranslate2, to
models/dualwhisper/<size>/final_checkpoint_ct2.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from benchmark import config
from systems.whisperseg import common as whisperseg


def training_command(size):
    """Return the command line that trains the model of one training size."""
    detector = whisperseg.model_dir("whisperseg", size).resolve()
    options = {
        "--initial_model_path": detector / "final_checkpoint",
        "--train_dataset_folder": config.prepared_dir(f"train_{size}_joined").resolve(),
        "--model_folder": config.DATA.resolve() / "models" / "dualwhisper" / str(size),
        "--max_num_epochs": 12,
        "--batch_size": 4,
        "--learning_rate": 3e-6,
        "--val_ratio": 0.1,
        "--validate_per_epoch": 1,
        "--save_per_epoch": 1,
        "--max_to_keep": 3,
        "--early_stop_patience": 4,
        "--ignore_cluster": 0,
        "--clear_cluster_codebook": 1,
        "--num_workers": 4,
        "--use_wandb": 0,
    }
    launcher = Path(__file__).resolve().parent / "whisperseg_training.py"
    command = [sys.executable, str(launcher)]
    for name, value in options.items():
        command += [name, str(value)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: environment variable CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    environment = dict(os.environ)
    # Transformers must not load TensorFlow: together with the libraries loaded during
    # validation it crashes the process.
    environment["USE_TF"] = "0"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    subprocess.run(
        training_command(arguments.size),
        cwd=whisperseg.WHISPERSEG,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
