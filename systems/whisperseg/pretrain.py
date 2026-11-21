"""Detection pretraining of WhisperSeg on the synthetic pretraining set.

    python -m systems.whisperseg.pretrain [--data DIR]

Starts from the released multispecies checkpoint, trains for one epoch on
prepared/whisperseg_pretraining/train and validates every 15,000 steps on
prepared/whisperseg_pretraining/val. The checkpoint with the best validation score is written to
models/whisperseg/pretraining/final_checkpoint.
"""
import argparse

from benchmark import config
from systems.whisperseg import common


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    pretraining_set = common.pretraining_set_dir().resolve()
    common.run_training([
        "--initial_model_path", common.RELEASED_CHECKPOINT,
        "--train_dataset_folder", str(pretraining_set / "train"),
        "--val_dataset_folder", str(pretraining_set / "val"),
        "--model_folder", str(common.pretrained_model_dir().resolve()),
        "--max_num_epochs", "1",
        "--validate_every", "15000",
        "--save_every", "15000",
    ])


if __name__ == "__main__":
    main()
