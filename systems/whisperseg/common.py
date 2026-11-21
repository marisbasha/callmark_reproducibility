"""Locations and training settings shared by the WhisperSeg scripts."""
import os
import subprocess
import sys
from pathlib import Path

from benchmark import config

# Clone of https://github.com/nianlonggu/WhisperSeg with whisperseg.patch applied (README.md).
WHISPERSEG = Path(__file__).resolve().parent / "WhisperSeg"

SYSTEMS = ("whisperseg", "whisperseg_no_pretraining")

# The released multispecies WhisperSeg checkpoint on the Hugging Face Hub.
RELEASED_CHECKPOINT = "nccratliri/whisperseg-large-ms"

# Arguments of WhisperSeg's train.py common to detection pretraining and fine-tuning. Every
# annotation is trained as the single class "Vocal" (ignore_cluster). The seed (66100), weight
# decay (0.01), warm-up (100 steps) and linear decay are the defaults of train.py.
TRAINING_ARGUMENTS = [
    "--batch_size", "4",
    "--learning_rate", "3e-6",
    "--max_to_keep", "3",
    "--early_stop_patience", "4",
    "--ignore_cluster", "1",
    "--clear_cluster_codebook", "1",
    "--num_workers", "4",
    "--use_wandb", "0",
]


def pretraining_set_dir():
    """Return the folder of the synthetic detection-pretraining set."""
    return config.DATA / "prepared" / "whisperseg_pretraining"


def pretrained_model_dir():
    """Return the folder of the detection-pretrained model."""
    return config.DATA / "models" / "whisperseg" / "pretraining"


def model_dir(system, size):
    """Return the folder of the model of one system fine-tuned at one training size."""
    return config.DATA / "models" / system / str(size)


def run_training(arguments):
    """Run WhisperSeg's train.py with `arguments` and stop if it fails."""
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "train.py"] + arguments + TRAINING_ARGUMENTS
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=WHISPERSEG, env=environment, check=True)
