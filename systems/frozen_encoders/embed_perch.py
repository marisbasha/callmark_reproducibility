"""Perch embeddings of the identification examples.

    python -m systems.frozen_encoders.embed_perch --data DIR

Writes embeddings/perch.npy with one row per identification example, all of train_500 and then
all of test in dataset order: the 1536-dimensional embedding of the vocalization at 32 kHz.
"""
import argparse

import numpy as np
from perch_hoplite.zoo import model_configs

from benchmark import config
from systems.frozen_encoders import clips

SAMPLING_RATE = 32000


def embed(rows):
    """Return the embedding matrix of `rows`, one row per example in the given order."""
    model = model_configs.load_model_by_name("perch_v2")
    embeddings = []
    for index, example in enumerate(rows):
        samples = clips.vocalization(example, SAMPLING_RATE)
        output = np.asarray(model.embed(samples).embeddings, dtype=np.float32)
        # The model returns one embedding per 5-s frame and channel; they are averaged.
        embeddings.append(output.reshape(-1, output.shape[-1]).mean(axis=0))
        if index % 500 == 0:
            print(f"{index}/{len(rows)}", flush=True)
    return np.stack(embeddings).astype(np.float32)


def main():
    """Embed every identification example and write the embedding file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()
    config.use_data_root(args.data)
    embeddings = embed(clips.embedding_examples())
    output_file = config.embedding_file("perch")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, embeddings)
    print(output_file, embeddings.shape)


if __name__ == "__main__":
    main()
