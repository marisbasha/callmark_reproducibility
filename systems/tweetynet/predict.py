"""Detect vocalizations in the test chunks with the trained TweetyNet of one training size.

    python -m systems.tweetynet.predict --data DIR --size 50

A vocalization is every run of 5-ms frames whose vocal probability is at least 0.5; no gaps
are filled and no short runs are removed. Writes predictions/tweetynet/detection_<size>.json.gz.
"""
import argparse

import torch

from benchmark import config
from systems.das import data
from systems.tweetynet import features


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = features.build_model().to(device)
    weights = torch.load(features.model_file(arguments.size), map_location=device)
    model.load_state_dict(weights)
    model.eval()

    rows = []
    for chunk_id, waveform in data.test_waveforms():
        log_energy, frame_seconds = features.spectrogram(waveform)
        probability = features.vocal_probability(model, log_energy, device)
        detected = data.vocalizations(probability, 1.0 / frame_seconds)
        rows.append({"chunk_id": chunk_id, "pred": detected})
        if len(rows) % 500 == 0:
            print(f"{len(rows)} chunks", flush=True)
    data.write_predictions("tweetynet", arguments.size, rows)


if __name__ == "__main__":
    main()
