"""One pretraining stage of WhisperID on xeno-canto vocalizations.

    python -m systems.whisperid.pretraining.pretrain --manifest FILE --base CHECKPOINT \
        --out DIR --epochs 2 --batch 8 --validation-fraction 0.005

Every line of the manifest is one vocalization with its taxon. A line with a background is
mixed into a stretch of benchmark training audio, optionally with a second vocalization of
another taxon; a line without background is shown in its own recording. The model sees a 5-s
window with the vocalization marked and generates the taxon name. The checkpoint with the
lowest validation loss is written to --out.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from systems.whisperid import features, model

SAMPLING_RATE = 32000
WINDOW_SAMPLES = int(features.WINDOW_SECONDS * SAMPLING_RATE)
SEED = 42
LEARNING_RATE = 1e-5
WORKERS = 10
# Level of the vocalization relative to the background, and of a second vocalization (dB).
SIGNAL_TO_BACKGROUND = (-5.0, 15.0)
SECOND_TO_BACKGROUND = (-8.0, 8.0)
# A vocalization longer than this share of the window is cut.
LONGEST_SHARE = 0.8


def load_audio(path, start, seconds):
    """Return `seconds` of a file from `start` at 32 kHz, padded with zeros; zeros on failure."""
    wanted = int(seconds * SAMPLING_RATE)
    try:
        samples, _ = librosa.load(
            path, sr=SAMPLING_RATE, offset=max(0.0, start), duration=seconds, mono=True
        )
    except Exception:
        return np.zeros(wanted, dtype=np.float32)
    if len(samples) < wanted:
        samples = np.concatenate([samples, np.zeros(wanted - len(samples), dtype=np.float32)])
    return samples[:wanted].astype(np.float32)


def level(samples):
    """Return the root mean square of a waveform."""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def window_in_recording(item, generator):
    """Return (waveform, onset, offset) of a vocalization inside its own recording.

    The window is centred on the vocalization and moved by up to one second."""
    half = features.WINDOW_SECONDS / 2
    centre = (item["onset"] + item["offset"]) / 2 + generator.uniform(-half * 0.4, half * 0.4)
    start = max(0.0, centre - half)
    try:
        waveform, _ = librosa.load(
            item["audio"],
            sr=SAMPLING_RATE,
            offset=start,
            duration=features.WINDOW_SECONDS,
            mono=True,
        )
    except Exception:
        waveform = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    if len(waveform) < WINDOW_SAMPLES:
        silence = np.zeros(WINDOW_SAMPLES - len(waveform), dtype=np.float32)
        waveform = np.concatenate([waveform, silence])
    return waveform, item["onset"] - start, item["offset"] - start


def mixture(item, generator):
    """Return (waveform, onset, offset) of a vocalization mixed into a background stretch.

    A background shorter than the window is repeated. The vocalization is placed at a random
    position and scaled to a random level relative to the background."""
    stretch = max(0.05, item["bg_end"] - item["bg_start"])
    stretch = min(stretch, features.WINDOW_SECONDS)
    background = load_audio(item["bg_audio"], item["bg_start"], stretch)
    if len(background) < WINDOW_SAMPLES:
        repeats = int(np.ceil(WINDOW_SAMPLES / max(len(background), 1)))
        background = np.tile(background, repeats)[:WINDOW_SAMPLES]

    longest = features.WINDOW_SECONDS * LONGEST_SHARE
    duration = min(item["offset"] - item["onset"], longest)
    vocalization = load_audio(item["audio"], item["onset"], duration)
    onset = generator.uniform(0.0, max(0.0, features.WINDOW_SECONDS - duration))
    first = int(onset * SAMPLING_RATE)

    mixed = background.copy()
    vocalization_level = level(vocalization)
    background_level = level(background)
    if vocalization_level > 0 and background_level > 0:
        decibels = generator.uniform(*SIGNAL_TO_BACKGROUND)
        gain = background_level * (10 ** (decibels / 20.0)) / vocalization_level
        mixed[first:first + len(vocalization)] += gain * vocalization
    elif vocalization_level > 0:
        mixed[first:first + len(vocalization)] += vocalization

    if "overlay_audio" in item:
        second_duration = min(item["overlay_offset"] - item["overlay_onset"], longest)
        second = load_audio(item["overlay_audio"], item["overlay_onset"], second_duration)
        second_level = level(second)
        if second_level > 0 and background_level > 0:
            decibels = generator.uniform(*SECOND_TO_BACKGROUND)
            gain = background_level * (10 ** (decibels / 20.0)) / second_level
            latest = max(0.0, features.WINDOW_SECONDS - second_duration)
            first = int(generator.uniform(0.0, latest) * SAMPLING_RATE)
            mixed[first:first + len(second)] += gain * second

    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed, onset, onset + duration


class Vocalizations(Dataset):
    """The marked spectrogram and the target tokens of every manifest line."""

    def __init__(self, items, tokenizer, seed):
        self.items = items
        self.tokenizer = tokenizer
        self.seed = seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        generator = random.Random(self.seed * 1_000_003 + index)
        if "bg_audio" in item:
            waveform, onset, offset = mixture(item, generator)
        else:
            waveform, onset, offset = window_in_recording(item, generator)
        spectrogram = features.spectrogram(waveform, SAMPLING_RATE)
        spectrogram = features.mark(spectrogram, onset, offset)
        decoder_input, labels = model.training_tokens(
            self.tokenizer, SAMPLING_RATE, item["species"]
        )
        return {
            "input_features": spectrogram,
            "decoder_input_ids": decoder_input,
            "labels": labels,
        }


def split_by_recording(items, validation_fraction):
    """Return (training items, validation items); whole recordings are held out."""
    generator = random.Random(SEED)
    by_recording = defaultdict(list)
    for item in items:
        by_recording[item["audio"]].append(item)
    recordings = sorted(by_recording)
    generator.shuffle(recordings)
    wanted = max(200, int(len(items) * validation_fraction))
    training = []
    validation = []
    for recording in recordings:
        if len(validation) < wanted:
            validation.extend(by_recording[recording])
        else:
            training.extend(by_recording[recording])
    generator.shuffle(training)
    generator.shuffle(validation)
    return training, validation


def loss_of(whisper, batch):
    """Return the training loss of one batch."""
    batch = {name: tensor.to(whisper.device) for name, tensor in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return whisper(**batch).loss


def main():
    """Run one pretraining stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base", required=True, help="checkpoint directory or model name")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, required=True)
    parser.add_argument("--mixture-seed", type=int, default=SEED,
                        help="seed of the training mixtures; a seed other than 42 also "
                             "seeds the batch order")
    args = parser.parse_args()

    with open(args.manifest) as handle:
        items = [json.loads(line) for line in handle]
    training, validation = split_by_recording(items, args.validation_fraction)
    print(f"{len(training)} training and {len(validation)} validation vocalizations", flush=True)

    whisper, tokenizer = model.load(args.base, torch.device("cuda"))
    optimizer = torch.optim.AdamW(whisper.parameters(), lr=LEARNING_RATE)
    join = model.collate(tokenizer.pad_token_id)
    training_loader = DataLoader(
        Vocalizations(training, tokenizer, args.mixture_seed),
        batch_size=args.batch, shuffle=True, num_workers=WORKERS, collate_fn=join, drop_last=True,
    )
    validation_loader = DataLoader(
        Vocalizations(validation, tokenizer, SEED),
        batch_size=args.batch, shuffle=False, num_workers=WORKERS, collate_fn=join,
    )
    if args.mixture_seed != SEED:
        torch.manual_seed(args.mixture_seed)

    output = Path(args.out)
    lowest_loss = float("inf")
    start = time.time()
    for epoch in range(args.epochs):
        whisper.train()
        total_loss = 0.0
        steps = 0
        for batch in training_loader:
            loss = loss_of(whisper, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(whisper.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss)
            steps += 1
            if steps % 200 == 0:
                minutes = (time.time() - start) / 60
                print(f"  epoch {epoch} step {steps}/{len(training_loader)} "
                      f"loss {total_loss / steps:.4f} ({minutes:.1f} min)", flush=True)
        whisper.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                validation_losses.append(float(loss_of(whisper, batch)))
        validation_loss = float(np.mean(validation_losses))
        print(f"epoch {epoch} loss {total_loss / max(steps, 1):.4f} "
              f"validation loss {validation_loss:.4f}", flush=True)
        if validation_loss < lowest_loss:
            lowest_loss = validation_loss
            whisper.save_pretrained(output)
            tokenizer.save_pretrained(output)
            print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
