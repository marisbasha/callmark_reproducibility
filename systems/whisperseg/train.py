"""Fine-tune WhisperSeg for detection at one training size.

    python -m systems.whisperseg.train --system whisperseg --size 50 [--data DIR]

`whisperseg` starts from the detection-pretrained model (pretrain.py),
`whisperseg_no_pretraining` from the released multispecies checkpoint. Trains on
prepared/train_<size>_joined for at most 12 epochs. After every epoch the model is validated on
a 10 % slice cut from the start or the end of every training file; training stops when the best
validation score is four epochs old. The best checkpoint is written to
models/<system>/<size>/final_checkpoint.
"""
import argparse

from benchmark import config
from systems.whisperseg import common


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--system", required=True, choices=common.SYSTEMS)
    parser.add_argument("--size", required=True, type=int, choices=config.SIZES)
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    if arguments.system == "whisperseg":
        initial_model = str(common.pretrained_model_dir().resolve() / "final_checkpoint")
    else:
        initial_model = common.RELEASED_CHECKPOINT
    training_set = config.prepared_dir(f"train_{arguments.size}_joined").resolve()
    model_folder = common.model_dir(arguments.system, arguments.size).resolve()
    common.run_training([
        "--initial_model_path", initial_model,
        "--train_dataset_folder", str(training_set),
        "--model_folder", str(model_folder),
        "--max_num_epochs", "12",
        "--val_ratio", "0.1",
        "--validate_per_epoch", "1",
        "--save_per_epoch", "1",
    ])


if __name__ == "__main__":
    main()
