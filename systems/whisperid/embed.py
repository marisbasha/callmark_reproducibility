"""Embeddings of a frozen encoder for the identification probes.

    python -m systems.whisperid.embed --data DIR --encoder whisperid_pretrained

Writes embeddings/<encoder>.npy with one row per identification example, all of train_500 and
then all of test in dataset order: the encoder output of the marked 5-s spectrogram, averaged
over the frames of the vocalization.
"""
import argparse
from collections import defaultdict

import numpy as np
import torch

from benchmark import config
from systems.whisperid import examples, features, model

BATCH = 16

# Encoders from the Hugging Face hub; every other encoder is models/whisperid/<directory>.
PUBLISHED = {
    "whisperseg": "nccratliri/whisperseg-large-ms",
    "whisper_large": "openai/whisper-large",
}
DIRECTORY = {
    "xeno_canto_only": "xeno_canto",
    "whisperid_bridge": "bridge",
    "whisperid_bridge_xcl1": "bridge_xcl",
    "whisperid_pretrained": "pretrained",
    "bridge_replay": "bridge_replay",
}


def checkpoint_of(encoder):
    """Return the checkpoint of a named encoder."""
    if encoder in PUBLISHED:
        return PUBLISHED[encoder]
    return str(config.DATA / "models" / "whisperid" / DIRECTORY[encoder])


def vocalization_frames(example):
    """Return (first, end) of the encoder frames that the vocalization covers."""
    frame_seconds = 2 * features.COLUMN_SECONDS
    first = max(0, int(example["onset_in_window"] / frame_seconds))
    end = int(np.ceil(example["offset_in_window"] / frame_seconds))
    end = min(features.ENCODER_FRAMES, end)
    return first, end


@torch.no_grad()
def embed(encoder, rows):
    """Return the embedding matrix of `rows`, one row per example in the given order."""
    by_rate = defaultdict(list)
    for index, example in enumerate(rows):
        by_rate[example["sampling_rate"]].append(index)
    embeddings = np.zeros((len(rows), encoder.config.d_model), dtype=np.float32)
    device = encoder.embed_positions.weight.device
    done = 0
    for sampling_rate in sorted(by_rate):
        indices = by_rate[sampling_rate]
        for start in range(0, len(indices), BATCH):
            batch = indices[start:start + BATCH]
            spectrograms = []
            for index in batch:
                waveform = examples.waveform(rows[index])
                spectrograms.append(features.example_features(rows[index], waveform))
            inputs = torch.from_numpy(np.stack(spectrograms)).to(device)
            hidden = encoder(inputs).last_hidden_state.float()
            for position, index in enumerate(batch):
                first, end = vocalization_frames(rows[index])
                if end > first:
                    frames = hidden[position, first:end]
                else:
                    frames = hidden[position]
                embeddings[index] = frames.mean(0).cpu().numpy()
            done += len(batch)
        print(f"{done}/{len(rows)}", flush=True)
    return embeddings


def main():
    """Embed every identification example with one encoder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--encoder", required=True, choices=sorted(PUBLISHED) + sorted(DIRECTORY))
    args = parser.parse_args()
    config.use_data_root(args.data)

    torch.backends.cuda.matmul.allow_tf32 = True
    whisper, _ = model.load(checkpoint_of(args.encoder), torch.device("cuda"))
    whisper.eval()
    rows = examples.rows("train_500") + examples.rows("test")
    embeddings = embed(whisper.model.encoder, rows)
    output_file = config.embedding_file(args.encoder)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, embeddings)
    print(f"wrote {output_file} {embeddings.shape}", flush=True)


if __name__ == "__main__":
    main()
