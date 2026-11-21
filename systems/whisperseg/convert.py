"""Convert a fine-tuned WhisperSeg model to CTranslate2 for prediction.

    python -m systems.whisperseg.convert --system whisperseg --size 50 [--data DIR]

Reads models/<system>/<size>/final_checkpoint, writes models/<system>/<size>/final_checkpoint_ct2
(weights as 8-bit integers, computation in 16-bit floating point). Run in the prediction
environment.
"""
import argparse
import subprocess
import sys

from benchmark import config
from systems.whisperseg import common


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--system", required=True, choices=common.SYSTEMS)
    parser.add_argument("--size", required=True, type=int, choices=config.SIZES)
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    model_folder = common.model_dir(arguments.system, arguments.size)
    command = [
        sys.executable, str(common.WHISPERSEG / "convert_hf_to_ct2.py"),
        "--model", str(model_folder / "final_checkpoint"),
        "--output_dir", str(model_folder / "final_checkpoint_ct2"),
        "--quantization", "int8_float16",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
