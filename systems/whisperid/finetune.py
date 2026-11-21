"""Fine-tune the pretrained WhisperID encoder-decoder on one training size.

    python -m systems.whisperid.finetune --data DIR --size 50

Starts from models/whisperid/pretrained and writes models/whisperid/finetuned_<size>: the
average of the weights after every epoch from the second epoch on. Training stops when the
validation macro-F1 of greedy decoding has not improved for eight epochs.
"""
import argparse
import json
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from benchmark import config
from systems.whisperid import examples, features, model

SEED = 42
MAX_EPOCHS = 50
PATIENCE = 8
BATCH = 8
LEARNING_RATE = 1e-5
WORKERS = 6
# The weights are averaged from the epoch with this index on (the first epoch has index 0).
FIRST_AVERAGED_EPOCH = 1


class TrainingExamples(Dataset):
    """The training examples of one sampling rate, augmented anew in every epoch."""

    def __init__(self, rows, tokenizer, seed):
        self.rows = rows
        self.tokenizer = tokenizer
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        example = self.rows[index]
        generator = np.random.default_rng(self.seed * 1_000_003 + index)
        waveform = examples.waveform(example)
        spectrogram = features.training_features(example, waveform, generator)
        decoder_input, labels = model.training_tokens(
            self.tokenizer, example["sampling_rate"], example["species"]
        )
        return {
            "input_features": spectrogram,
            "decoder_input_ids": decoder_input,
            "labels": labels,
        }


def by_sampling_rate(rows):
    """Return the examples grouped by sampling rate, in order of first occurrence."""
    groups = defaultdict(list)
    for example in rows:
        groups[example["sampling_rate"]].append(example)
    return groups


def epoch_loaders(rows, tokenizer, seed):
    """Return one shuffled loader per sampling rate for one epoch."""
    loaders = []
    for group in by_sampling_rate(rows).values():
        loader = DataLoader(
            TrainingExamples(group, tokenizer, seed),
            batch_size=BATCH,
            shuffle=True,
            num_workers=WORKERS,
            collate_fn=model.collate(tokenizer.pad_token_id),
        )
        loaders.append(loader)
    return loaders


def train_epoch(whisper, optimizer, loaders, seed):
    """Train one epoch, drawing every batch from a randomly chosen sampling rate."""
    whisper.train()
    iterators = [iter(loader) for loader in loaders]
    generator = random.Random(seed)
    total_loss = 0.0
    steps = 0
    while iterators:
        iterator = generator.choice(iterators)
        try:
            batch = next(iterator)
        except StopIteration:
            iterators.remove(iterator)
            continue
        batch = {name: tensor.to(whisper.device) for name, tensor in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = whisper(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(whisper.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss)
        steps += 1
        if steps % 100 == 0:
            print(f"  step {steps} loss {total_loss / steps:.4f}", flush=True)
    return total_loss / max(steps, 1)


def scored_taxa():
    """Return the taxa of the dataset that are marked as scorable."""
    taxa = set()
    for example in examples.rows("test", with_audio=False):
        if example["scorable"]:
            taxa.add(example["species"])
    return taxa


def macro_f1(truth, predicted, taxa):
    """Return the mean F1 over the taxa of `taxa` that occur in `truth`."""
    correct = defaultdict(int)
    false_positive = defaultdict(int)
    missed = defaultdict(int)
    for true_taxon, predicted_taxon in zip(truth, predicted):
        if true_taxon == predicted_taxon:
            correct[true_taxon] += 1
        else:
            missed[true_taxon] += 1
            false_positive[predicted_taxon] += 1
    scores = []
    for taxon in taxa:
        if correct[taxon] + missed[taxon] == 0:
            continue
        denominator = 2 * correct[taxon] + false_positive[taxon] + missed[taxon]
        scores.append(round(2 * correct[taxon] / denominator, 4))
    return float(np.mean(scores))


def validate(whisper, tokenizer, rows, taxa):
    """Return the macro-F1 of greedily generated names on the validation examples."""
    whisper.eval()
    truth = []
    predicted = []
    for sampling_rate, group in by_sampling_rate(rows).items():
        for start in range(0, len(group), BATCH):
            batch = group[start:start + BATCH]
            spectrograms = []
            for example in batch:
                waveform = examples.waveform(example)
                spectrograms.append(features.example_features(example, waveform))
            inputs = torch.from_numpy(np.stack(spectrograms))
            predicted.extend(model.generate_taxa(whisper, tokenizer, inputs, sampling_rate))
            truth.extend(example["species"] for example in batch)
    return macro_f1(truth, predicted, taxa)


def update_average(average, count, whisper):
    """Return the running mean of the weights after including the current weights."""
    current = {name: value.detach().float().cpu() for name, value in whisper.state_dict().items()}
    if average is None:
        return current, 1
    count += 1
    for name in average:
        average[name] += (current[name] - average[name]) / count
    return average, count


def main():
    """Fine-tune on one training size and save the weight average."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    args = parser.parse_args()
    config.use_data_root(args.data)
    models = config.DATA / "models" / "whisperid"
    output = models / f"finetuned_{args.size}"

    torch.manual_seed(SEED)
    taxa = scored_taxa()
    training, validation = examples.split_validation(examples.rows(f"train_{args.size}"))
    print(f"{len(training)} training and {len(validation)} validation examples", flush=True)
    whisper, tokenizer = model.load(str(models / "pretrained"), torch.device("cuda"))
    optimizer = torch.optim.AdamW(whisper.parameters(), lr=LEARNING_RATE)

    best_score = -1.0
    best_epoch = 0
    average = None
    averaged_epochs = 0
    log = []
    start = time.time()
    for epoch in range(MAX_EPOCHS):
        loaders = epoch_loaders(training, tokenizer, SEED + epoch)
        loss = train_epoch(whisper, optimizer, loaders, SEED + epoch)
        score = validate(whisper, tokenizer, validation, taxa)
        minutes = (time.time() - start) / 60
        print(f"epoch {epoch} loss {loss:.4f} validation macro-F1 {score:.4f} "
              f"({minutes:.1f} min)", flush=True)
        log.append({"epoch": epoch, "loss": loss, "validation_macro_f1": round(score, 4)})
        if epoch >= FIRST_AVERAGED_EPOCH:
            average, averaged_epochs = update_average(average, averaged_epochs, whisper)
        if score > best_score:
            best_score = score
            best_epoch = epoch
        elif epoch - best_epoch >= PATIENCE:
            break

    weights = whisper.state_dict()
    whisper.load_state_dict({name: average[name].to(weights[name].dtype) for name in weights})
    whisper.save_pretrained(output)
    tokenizer.save_pretrained(output)
    with open(output / "training_log.json", "w") as handle:
        json.dump({"epochs": log, "averaged_epochs": averaged_epochs}, handle, indent=1)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
