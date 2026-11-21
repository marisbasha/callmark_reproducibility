"""Train DAS (Deep Audio Segmenter) on the joined training files of one training size.

    python -m systems.das.train --data DIR --size 50

One model for all corpora on the waveform at 32 kHz, two classes per sample (noise, vocal).
A tenth of the training files of every corpus is held out; DAS keeps the model of the epoch
with the lowest validation loss. Writes models/das/<size>/das_model.h5 and das_params.yaml.
"""
import argparse
import shutil
import time

import numpy as np

from benchmark import config
from systems.das import data, network

EPOCHS = 40
BATCH_SIZE = 64
INPUT_SAMPLES = 4096
FILTERS = 32
KERNEL_SIZE = 16
CONVOLUTION_BLOCKS = 4


def sample_arrays(files):
    """Return the audio and the labels of the files, concatenated in the given order.

    The audio has shape (samples, 1), the labels (samples, 2) with columns noise and vocal."""
    waveforms = []
    vocal_tracks = []
    for file in files:
        waveform = data.training_waveform(file)
        vocal = np.zeros(len(waveform), dtype=np.float32)
        for onset, offset in zip(file["onsets"], file["offsets"]):
            start = max(0, int(round(float(onset) * data.SAMPLE_RATE)))
            end = min(len(waveform), int(round(float(offset) * data.SAMPLE_RATE)))
            if end > start:
                vocal[start:end] = 1.0
        waveforms.append(waveform)
        vocal_tracks.append(vocal)
    audio = np.concatenate(waveforms).astype(np.float32)[:, None]
    vocal = np.concatenate(vocal_tracks)
    labels = np.stack([1.0 - vocal, vocal], axis=1).astype(np.float32)
    return audio, labels


def write_training_arrays(directory, size):
    """Write the training and validation arrays in the directory layout that DAS reads."""
    if directory.exists():
        shutil.rmtree(directory)
    files = data.training_files(size)
    training, validation = data.split_training_files(files)
    for name, part in (("train", training), ("val", validation)):
        audio, labels = sample_arrays(part)
        (directory / name).mkdir(parents=True)
        np.save(directory / name / "x.npy", audio)
        np.save(directory / name / "y.npy", labels)
        hours = len(audio) / data.SAMPLE_RATE / 3600
        print(f"{name}: {len(part)} files, {hours:.2f} h", flush=True)
    attributes = {
        "samplerate_x_Hz": data.SAMPLE_RATE,
        "samplerate_y_Hz": data.SAMPLE_RATE,
        "class_names": ["noise", "vocal"],
        "class_types": ["segment", "segment"],
    }
    np.save(directory / "attrs.npy", attributes, allow_pickle=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    model_dir = network.model_dir(arguments.size)
    model_dir.mkdir(parents=True, exist_ok=True)
    array_dir = model_dir / "training.npy"
    write_training_arrays(array_dir, arguments.size)

    network.prepare_keras()
    import das.train
    start = time.time()
    das.train.train(
        data_dir=str(array_dir),
        save_dir=str(model_dir),
        save_name=network.MODEL_NAME,
        model_name="tcn",
        nb_hist=INPUT_SAMPLES,
        nb_filters=FILTERS,
        kernel_size=KERNEL_SIZE,
        nb_conv=CONVOLUTION_BLOCKS,
        use_separable=[True] * CONVOLUTION_BLOCKS,
        nb_epoch=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        log_messages=False,
        reduce_lr=True,
    )
    print(f"trained in {(time.time() - start) / 60:.1f} min")
    shutil.rmtree(array_dir)


if __name__ == "__main__":
    main()
