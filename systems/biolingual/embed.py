"""BioLingual embeddings of the identification examples.

    python -m systems.biolingual.embed --data DIR

Writes embeddings/biolingual.npy with one row per identification example, all of train_500 and
then all of test in dataset order: the 512-dimensional audio embedding of the vocalization at
48 kHz from the unchanged published model.
"""
import argparse

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from benchmark import config
from systems.biolingual import common
from systems.frozen_encoders import clips

BATCH = 16


@torch.no_grad()
def embed(rows):
    """Return the embedding matrix of `rows`, one row per example in the given order."""
    model = ClapModel.from_pretrained(common.MODEL_NAME).cuda().eval()
    processor = ClapProcessor.from_pretrained(common.MODEL_NAME)
    embeddings = []
    for start in range(0, len(rows), BATCH):
        waveforms = []
        for example in rows[start:start + BATCH]:
            waveforms.append(clips.vocalization(example, common.SAMPLING_RATE))
        inputs = processor(
            audios=waveforms,
            sampling_rate=common.SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        features = model.get_audio_features(**inputs.to("cuda"))
        embeddings.append(features.cpu().numpy().astype(np.float32))
        if start % (BATCH * 20) == 0:
            print(f"{start}/{len(rows)}", flush=True)
    return np.concatenate(embeddings)


def main():
    """Embed every identification example and write the embedding file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()
    config.use_data_root(args.data)
    embeddings = embed(clips.embedding_examples())
    output_file = config.embedding_file("biolingual")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, embeddings)
    print(output_file, embeddings.shape)


if __name__ == "__main__":
    main()
