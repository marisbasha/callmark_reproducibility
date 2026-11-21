"""BirdNET embeddings of the identification examples.

    python -m systems.frozen_encoders.embed_birdnet --data DIR

Writes embeddings/birdnet.npy with one row per identification example, all of train_500 and then
all of test in dataset order: the 1024-dimensional embedding of one 3-s window at 48 kHz with
the vocalization in its centre.
"""
import argparse

import numpy as np
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer
from tensorflow import lite

from benchmark import config
from systems.frozen_encoders import clips

SAMPLING_RATE = 48000
WINDOW_SECONDS = 3.0


def centred_window(samples):
    """Return one analysis window with the vocalization in its centre.

    A shorter vocalization is padded with zeros, of a longer one the central 3 s are kept."""
    length = int(round(WINDOW_SECONDS * SAMPLING_RATE))
    if len(samples) >= length:
        first = (len(samples) - length) // 2
        return samples[first:first + length]
    window = np.zeros(length, dtype=np.float32)
    first = (length - len(samples)) // 2
    window[first:first + len(samples)] = samples
    return window


def load_analyzer():
    """Return the BirdNET analyzer with an interpreter that keeps intermediate tensors.

    The embedding is an intermediate tensor of the network, which the interpreter of birdnetlib
    discards."""
    analyzer = Analyzer()
    analyzer.interpreter = lite.Interpreter(
        model_path=analyzer.model_path,
        num_threads=1,
        experimental_preserve_all_tensors=True,
    )
    analyzer.interpreter.allocate_tensors()
    analyzer.input_details = analyzer.interpreter.get_input_details()
    analyzer.output_details = analyzer.interpreter.get_output_details()
    analyzer.input_layer_index = analyzer.input_details[0]["index"]
    analyzer.output_layer_index = analyzer.output_details[0]["index"]
    return analyzer


def embed(rows):
    """Return the embedding matrix of `rows`, one row per example in the given order."""
    analyzer = load_analyzer()
    embeddings = []
    for index, example in enumerate(rows):
        window = centred_window(clips.vocalization(example, SAMPLING_RATE))
        recording = RecordingBuffer(analyzer, window, SAMPLING_RATE)
        recording.extract_embeddings()
        embeddings.append(np.asarray(recording.embeddings[0]["embeddings"], dtype=np.float32))
        if index % 1000 == 0:
            print(f"{index}/{len(rows)}", flush=True)
    return np.stack(embeddings).astype(np.float32)


def main():
    """Embed every identification example and write the embedding file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()
    config.use_data_root(args.data)
    embeddings = embed(clips.embedding_examples())
    output_file = config.embedding_file("birdnet")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, embeddings)
    print(output_file, embeddings.shape)


if __name__ == "__main__":
    main()
