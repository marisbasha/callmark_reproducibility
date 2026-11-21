"""Frozen-encoder probes for identification.

    python -m benchmark.probe [--data DIR] [--encoders perch birdnet ...]

For every encoder, embeddings/<encoder>.npy holds one row per identification example: all of
train_500, then all of test, in dataset order. A 512-unit MLP is fitted on the L2-normalised
embeddings of the training examples of each training size and its test predictions are written to
predictions/probe_<encoder>/identification_<size>.json.
"""
import argparse

import numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from benchmark import config, data

ENCODERS = (
    "whisperseg",
    "whisper_large",
    "whisperid_bridge",
    "whisperid_bridge_xcl1",
    "whisperid_pretrained",
    "xeno_canto_only",
    "bridge_replay",
    "perch",
    "birdnet",
    "biolingual",
)


def normalized(embeddings):
    """Return the embeddings scaled to unit L2 norm, row by row."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


def fit(encoder):
    """Fit the probe of one encoder at every training size and write its test predictions."""
    train = data.table("classification", "train_500", ("species", "fewshot_rank"))
    test_ids, _, _ = data.identification_test()
    embeddings = normalized(np.load(config.embedding_file(encoder)))
    train_embeddings = embeddings[:len(train)]
    test_embeddings = embeddings[len(train):]
    taxa = np.array([row["species"] for row in train])
    rank = np.array([row["fewshot_rank"] for row in train])
    for size in config.SIZES:
        # The training set of a given size holds the examples with fewshot_rank below that size.
        in_training_set = rank < size
        probe = MLPClassifier(
            hidden_layer_sizes=(512,),
            alpha=1e-4,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=42,
        )
        # A single thread makes the fit reproducible.
        with threadpool_limits(limits=1):
            probe.fit(train_embeddings[in_training_set], taxa[in_training_set])
            predicted = probe.predict(test_embeddings)
        output_file = config.prediction_file(f"probe_{encoder}", "identification", size)
        data.write_json(output_file, dict(zip(test_ids, predicted.tolist())))
        print(encoder, size, flush=True)


def main():
    """Fit the probes of the encoders given on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--encoders", nargs="+", default=ENCODERS)
    args = parser.parse_args()
    config.use_data_root(args.data)
    for encoder in args.encoders:
        fit(encoder)


if __name__ == "__main__":
    main()
