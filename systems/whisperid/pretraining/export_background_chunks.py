"""Write the training chunks that supply the background audio of the pretraining mixtures.

    python -m systems.whisperid.pretraining.export_background_chunks --data DIR

Writes whisperid/background_chunks/<chunk_id>.wav and .json (annotated onsets and offsets) for
the 500-size detection training chunks of the four soundscape and field corpora.
"""
import argparse
import io
import json

import pyarrow.parquet as pq
import soundfile

from benchmark import config

BACKGROUND_CORPORA = ("enabirds", "sne_full", "nbm", "wytham")


def main():
    """Export the chunks of the background corpora."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()
    config.use_data_root(args.data)
    output = config.DATA / "whisperid" / "background_chunks"
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((config.dataset_dir() / "segmentation").glob("train_500-*.parquet"))
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                if row["dataset"] not in BACKGROUND_CORPORA:
                    continue
                encoded = io.BytesIO(row["audio"]["bytes"])
                waveform, sampling_rate = soundfile.read(encoded, dtype="float32")
                name = row["chunk_id"]
                soundfile.write(output / f"{name}.wav", waveform, sampling_rate)
                labels = {"onset": row["onsets"], "offset": row["offsets"]}
                with open(output / f"{name}.json", "w") as handle:
                    json.dump(labels, handle)
        print(file.name, flush=True)


if __name__ == "__main__":
    main()
