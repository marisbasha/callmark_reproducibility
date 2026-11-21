"""Fine-tune BirdCODE on the detection training windows or on the identification clips.

    python -m systems.birdcode.train --data DIR --model detector --size 500
    python -m systems.birdcode.train --data DIR --model identifier --size 500

detector: trained on prepared/train_<size>; used for detection and Detection+Identification.
identifier: trained on the clips of prepare_identification_clips.py; used for identification.
Both write models/birdcode/<model>_<size>/best_model.pt, labels.txt and train_log.json.
"""
import argparse
import json
import math
import random
import time

import numpy
import torch

from benchmark import config
from systems.birdcode import birdcode

# Corpora that are part of the training material but not of the benchmark.
CORPORA_NOT_IN_BENCHMARK = ("right_whale", "spiders", config.DUPLICATE_CORPUS)

UNIDENTIFIED = "unknown"
SEED = 0
HOLDOUT_SEED = 42
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
GRADIENT_NORM_LIMIT = 1.0
MAXIMUM_EPOCHS = 100
PATIENCE = 5
NEW_PROTOTYPE_NOISE = 0.01


def annotation_files(model_name, size):
    """Return the annotation files of the training set of one model, sorted by name."""
    if model_name == "identifier":
        return sorted((birdcode.identification_clip_dir(size) / "clips").glob("*.json"))
    files = []
    for file in sorted(config.prepared_dir(f"train_{size}").glob("*.json")):
        if file.name.split("__")[0] not in CORPORA_NOT_IN_BENCHMARK:
            files.append(file)
    return files


def load_recordings(model_name, size, model):
    """Read the training recordings at the sampling rate of the model, with their annotations.

    The detector leaves out the corpora sampled above 64 kHz."""
    recordings = []
    for file in annotation_files(model_name, size):
        annotation = birdcode.read_json(file)
        too_high = annotation["sr"] > birdcode.HIGHEST_SAMPLING_RATE
        if model_name == "detector" and too_high:
            continue
        recordings.append({
            "name": file.stem,
            "waveform": birdcode.load_audio(file.with_suffix(".wav"), model),
            "onsets": annotation["onset"],
            "offsets": annotation["offset"],
            "taxa": annotation["cluster"],
        })
    return recordings


def taxa_of(recordings):
    """Return the sorted taxa of the identified vocalizations."""
    taxa = set()
    for recording in recordings:
        for taxon in recording["taxa"]:
            if taxon != UNIDENTIFIED:
                taxa.add(taxon)
    return sorted(taxa)


def restrict_to_taxa(model, taxa):
    """Reduce the output classes of the released model to `taxa` and return the added taxa.

    A taxon the released model knows keeps its prototypes, class weights and bias. Any other
    taxon starts from the prototypes of a randomly drawn known class with added noise, the
    mean class weights and the mean bias."""
    known = list(model.labels)
    per_class = model.classifier.features_per_output_class
    old_prototypes = model.encoder.prototype_vectors.data
    old_weights = model.classifier.weight.data
    old_biases = model.classifier.bias.data
    generator = torch.Generator().manual_seed(SEED)
    prototypes = []
    weights = []
    biases = []
    added = []
    for taxon in taxa:
        if taxon in known:
            index = known.index(taxon)
            prototypes.append(old_prototypes[index * per_class:(index + 1) * per_class])
            weights.append(old_weights[index])
            biases.append(old_biases[index])
        else:
            index = int(torch.randint(len(known), (1,), generator=generator))
            template = old_prototypes[index * per_class:(index + 1) * per_class]
            noise = torch.randn(template.shape, generator=generator)
            prototypes.append(template + NEW_PROTOTYPE_NOISE * noise)
            weights.append(old_weights.mean(0))
            biases.append(old_biases.mean())
            added.append(taxon)
    model.encoder.prototype_vectors = torch.nn.Parameter(torch.cat(prototypes))
    model.classifier = birdcode.PrototypeClassifier(per_class * len(taxa), len(taxa))
    model.classifier.weight.data = torch.stack(weights).clone()
    model.classifier.bias.data = torch.stack(biases).clone()
    model.labels = list(taxa)
    return added


def split_recordings(model_name, size, recordings):
    """Return the indices of the training and of the validation recordings.

    detector: a random tenth of the recordings is held out. identifier: the clips listed by
    prepare_identification_clips.py are held out."""
    if model_name == "identifier":
        listed = birdcode.identification_clip_dir(size) / "validation.txt"
        held_out = set(listed.read_text().split())
        training = []
        validation = []
        for index, recording in enumerate(recordings):
            if recording["name"] in held_out:
                validation.append(index)
            else:
                training.append(index)
        return training, validation
    order = list(range(len(recordings)))
    random.Random(HOLDOUT_SEED).shuffle(order)
    validation_count = max(1, len(order) // 10)
    return order[validation_count:], order[:validation_count]


def crop(recording, taxa, model, start_sample=None):
    """Return one window of a recording with its frame targets and loss mask.

    The window has the input length of the model (5 s) and starts at `start_sample`, or at a
    random sample. A frame is positive for a taxon when a vocalization of that taxon covers at
    least half of the frame or at least half of the vocalization lies in the frame. Frames
    touched by an unidentified vocalization carry no loss unless they are positive for a taxon.
    """
    frame_rate = model.frame_rate
    window_seconds = model.window_duration
    frame_count = int(round(window_seconds * frame_rate))
    window_samples = int(window_seconds * model.sample_rate)
    if start_sample is None:
        start_sample = random.randint(0, max(0, len(recording["waveform"]) - window_samples))
    waveform = recording["waveform"][start_sample:start_sample + window_samples]
    waveform = numpy.pad(waveform, (0, window_samples - len(waveform)))

    start = start_sample / model.sample_rate
    frame_edges = numpy.arange(frame_count + 1) / frame_rate + start
    target = numpy.zeros((frame_count, len(taxa)), numpy.float32)
    unidentified = numpy.zeros(frame_count, bool)
    for onset, offset, taxon in zip(recording["onsets"], recording["offsets"], recording["taxa"]):
        if offset <= start or onset >= start + window_seconds:
            continue
        overlap = numpy.minimum(frame_edges[1:], offset) - numpy.maximum(frame_edges[:-1], onset)
        overlap = numpy.clip(overlap, 0, None)
        if taxon == UNIDENTIFIED:
            unidentified |= overlap > 0
        else:
            covered = (overlap >= 0.5 / frame_rate) | (overlap >= 0.5 * (offset - onset))
            target[covered, taxa.index(taxon)] = 1
    mask = numpy.ones_like(target)
    mask[unidentified & (target.max(1) == 0)] = 0
    return waveform, target, mask


def batches(examples):
    """Yield the examples as batches of (waveforms, targets, masks) on the GPU."""
    for first in range(0, len(examples), BATCH_SIZE):
        batch = examples[first:first + BATCH_SIZE]
        tensors = []
        for part in range(3):
            stacked = numpy.stack([example[part] for example in batch])
            tensors.append(torch.tensor(stacked).to("cuda"))
        yield tensors


def masked_loss(model, waveforms, targets, masks):
    """Return the binary cross-entropy averaged over the frames and taxa that carry loss."""
    frame_count = targets.shape[1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(waveforms)[:, :frame_count].float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * masks).sum() / masks.sum().clamp(min=1)


def train_one_epoch(model, optimizer, examples):
    """Train on every example once and return the mean batch loss."""
    model.train()
    losses = []
    for waveforms, targets, masks in batches(examples):
        loss = masked_loss(model, waveforms, targets, masks)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM_LIMIT)
        optimizer.step()
        losses.append(loss.item())
    return float(numpy.mean(losses))


def validation_loss(model, examples):
    """Return the mean batch loss of the validation windows."""
    model.eval()
    losses = []
    with torch.no_grad():
        for waveforms, targets, masks in batches(examples):
            losses.append(masked_loss(model, waveforms, targets, masks).item())
    return float(numpy.mean(losses))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data root (default: environment variable CALLMARK_DATA)")
    parser.add_argument("--model", required=True, choices=("detector", "identifier"))
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    args = parser.parse_args()
    config.use_data_root(args.data)

    torch.manual_seed(SEED)
    random.seed(SEED)
    numpy.random.seed(SEED)
    out_dir = birdcode.model_dir(args.model, args.size)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = birdcode.load_pretrained()
    recordings = load_recordings(args.model, args.size, model)
    taxa = taxa_of(recordings)
    added_taxa = restrict_to_taxa(model, taxa)
    training, validation = split_recordings(args.model, args.size, recordings)
    print(f"{len(training)} training and {len(validation)} validation recordings, "
          f"{len(taxa)} taxa, added: {added_taxa}", flush=True)

    model = model.to("cuda").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Validation windows are fixed: the first 5 s of every validation recording.
    validation_windows = [crop(recordings[index], taxa, model, 0) for index in validation]

    best_loss = math.inf
    epochs_without_improvement = 0
    epochs = []
    for epoch in range(MAXIMUM_EPOCHS):
        started = time.time()
        random.shuffle(training)
        training_windows = [crop(recordings[index], taxa, model) for index in training]
        training_loss = train_one_epoch(model, optimizer, training_windows)
        loss = validation_loss(model, validation_windows)
        epochs.append({
            "epoch": epoch,
            "train_loss": training_loss,
            "val_loss": loss,
            "minutes": (time.time() - started) / 60,
        })
        print(epochs[-1], flush=True)
        if loss < best_loss:
            best_loss = loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            (out_dir / "labels.txt").write_text("\n".join(taxa) + "\n")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    log = {
        "taxa": taxa,
        "added_taxa": added_taxa,
        "training_recordings": len(training),
        "validation_recordings": len(validation),
        "best_validation_loss": best_loss,
        "epochs": epochs,
    }
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    print("best validation loss", best_loss)


if __name__ == "__main__":
    main()
