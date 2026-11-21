"""Fine-tune BioLingual for identification.

    python -m systems.biolingual.train --data DIR --size 500

All weights of the published model are trained together with a linear layer over the taxa of
the dataset, on the vocalizations of one training size. A tenth of the recordings of every
taxon is held out; after every epoch the macro-F1 on the held-out examples is computed and the
best model is kept as models/biolingual/train_<size>.pt.
"""
import os

# The feature extraction in the data-loading processes otherwise takes every processor core.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
from collections import defaultdict

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from benchmark import config
from systems.biolingual import common
from systems.frozen_encoders import clips

EPOCHS = 50
PATIENCE = 8
BATCH = 16
ENCODER_LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 1e-3
WORKERS = 3


def loader(examples, taxon_names, processor, training):
    """Return the data loader of a list of examples; training examples are augmented."""
    return torch.utils.data.DataLoader(
        common.Clips(examples, taxon_names, augment=training),
        batch_size=BATCH,
        shuffle=training,
        drop_last=training,
        num_workers=WORKERS,
        collate_fn=common.Collate(processor),
        pin_memory=True,
        worker_init_fn=common.limit_worker_threads,
        persistent_workers=True,
    )


@torch.no_grad()
def predicted_classes(model, batches):
    """Return (reference classes, predicted classes) over all batches of a loader."""
    model.eval()
    reference = []
    predicted = []
    for inputs, classes in batches:
        inputs = {name: value.cuda() for name, value in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted.extend(model(inputs).argmax(-1).tolist())
        reference.extend(classes.tolist())
    model.train()
    return reference, predicted


def macro_f1(reference, predicted, scored_classes):
    """Return the mean F1 over the scored classes that occur in the reference.

    The F1 of every class is rounded to four decimals before averaging."""
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    for reference_class, predicted_class in zip(reference, predicted):
        if reference_class == predicted_class:
            true_positives[reference_class] += 1
        else:
            false_negatives[reference_class] += 1
            false_positives[predicted_class] += 1
    scores = []
    for scored_class in scored_classes:
        hits = true_positives[scored_class]
        if hits + false_negatives[scored_class] == 0:
            continue
        if hits == 0:
            scores.append(0.0)
            continue
        precision = hits / (hits + false_positives[scored_class])
        recall = hits / (hits + false_negatives[scored_class])
        scores.append(round(2 * precision * recall / (precision + recall), 4))
    return float(np.mean(scores))


def train(size):
    """Fine-tune on one training size and write the model with the best validation macro-F1."""
    torch.manual_seed(common.SEED)
    torch.set_num_threads(1)
    taxon_names = common.taxa()
    scored = common.validation_taxa()
    scored_classes = [index for index, name in enumerate(taxon_names) if name in scored]
    training, validation = common.split_validation(clips.examples(f"train_{size}"))
    print(f"training {len(training)}, validation {len(validation)}", flush=True)

    processor = ClapProcessor.from_pretrained(common.MODEL_NAME)
    clap = ClapModel.from_pretrained(common.MODEL_NAME)
    model = common.Classifier(clap, len(taxon_names)).cuda()
    model.train()
    training_batches = loader(training, taxon_names, processor, training=True)
    validation_batches = loader(validation, taxon_names, processor, training=False)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.clap.parameters(), "lr": ENCODER_LEARNING_RATE},
            {"params": model.head.parameters(), "lr": HEAD_LEARNING_RATE},
        ],
        weight_decay=0.01,
    )
    # The learning-rate cycle spans all 50 epochs, also when training stops earlier.
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[ENCODER_LEARNING_RATE, HEAD_LEARNING_RATE],
        total_steps=len(training_batches) * EPOCHS,
        pct_start=0.1,
    )
    loss_function = torch.nn.CrossEntropyLoss()
    output_file = common.model_file(size)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_epoch = -1
    for epoch in range(EPOCHS):
        for step, (inputs, classes) in enumerate(training_batches):
            inputs = {name: value.cuda(non_blocking=True) for name, value in inputs.items()}
            classes = classes.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_function(model(inputs), classes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {float(loss):.4f}", flush=True)
        reference, predicted = predicted_classes(model, validation_batches)
        score = macro_f1(reference, predicted, scored_classes)
        print(f"epoch {epoch} validation macro-F1 {score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), output_file)
        elif epoch - best_epoch >= PATIENCE:
            break
    print(f"best epoch {best_epoch}, validation macro-F1 {best_score:.4f}: {output_file}")


def main():
    """Fine-tune the model of the training size given on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    args = parser.parse_args()
    config.use_data_root(args.data)
    train(args.size)


if __name__ == "__main__":
    main()
