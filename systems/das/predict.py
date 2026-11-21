"""Detect vocalizations in the test chunks with the trained DAS model of one training size.

    python -m systems.das.predict --data DIR --size 50

Every chunk is resampled to 32 kHz and passed through the model on its own. A vocalization is
every run of samples whose vocal probability is at least 0.5; no gaps are filled and no short
runs are removed. Writes predictions/das/detection_<size>.json.gz.
"""
import argparse

import numpy as np

from benchmark import config
from systems.das import data, network


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    network.prepare_keras()
    import das.predict
    import das.utils
    model, parameters = das.utils.load_model_and_params(network.model_prefix(arguments.size))

    rows = []
    for chunk_id, waveform in data.test_waveforms():
        samples = np.ascontiguousarray(waveform.astype(np.float32)[:, None])
        probabilities = das.predict.predict_probabilities(
            samples, model=model, params=parameters, verbose=0, prepend_data_padding=True
        )
        vocal_probability = np.asarray(probabilities)[:, 1].astype(np.float32)
        detected = data.vocalizations(vocal_probability, float(data.SAMPLE_RATE))
        rows.append({"chunk_id": chunk_id, "pred": detected})
        if len(rows) % 500 == 0:
            print(f"{len(rows)} chunks", flush=True)
    data.write_predictions("das", arguments.size, rows)


if __name__ == "__main__":
    main()
