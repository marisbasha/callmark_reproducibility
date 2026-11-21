"""Fine-tune the LoRA adapters of NatureLM-audio for identification.

    python -m systems.naturelm.train --data DIR --size 500

The model is given the vocalization at 16 kHz and the question PROMPT, and is trained to answer
with the taxon. Only the LoRA adapters of the language model are trained. A tenth of the
recordings of every taxon is held out; the model with the lowest validation loss is written to
models/naturelm/train_<size>/.
"""
import argparse

import torch
from NatureLM.models import NatureLM

from benchmark import config
from systems.biolingual import common
from systems.frozen_encoders import clips

MODEL_NAME = "EarthSpeciesProject/NatureLM-audio"
SAMPLING_RATE = 16000
# The audio encoder of the model fails on clips shorter than half a second.
MINIMUM_SECONDS = 0.5
PROMPT = "What is the scientific name for the focal species in the audio?"
TASK = "sci-simple-detection"

EPOCHS = 8
BATCH = 4
BATCHES_PER_STEP = 8
LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
VALIDATIONS = 12
VALIDATION_BATCHES = 40
PATIENCE = 4
WORKERS = 8


class Clips(torch.utils.data.Dataset):
    """Vocalizations at 16 kHz with their taxon as the answer."""

    def __init__(self, examples, augment):
        self.examples = examples
        self.augment = augment

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        samples = clips.vocalization(example, SAMPLING_RATE, MINIMUM_SECONDS)
        if self.augment:
            samples = common.augmented(samples, common.item_generator(index))
        return torch.from_numpy(samples), example["species"]


def collate(batch):
    """Pad the clips of a batch to equal length and build the input of the model."""
    waveforms = [waveform for waveform, _ in batch]
    padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    padding = torch.zeros(padded.shape, dtype=torch.bool)
    for index, waveform in enumerate(waveforms):
        padding[index, len(waveform):] = True
    return {
        "raw_wav": padded,
        "padding_mask": padding,
        "audio_chunk_sizes": [1] * len(batch),
        "text": [taxon for _, taxon in batch],
        "prompt": [PROMPT] * len(batch),
        "task": [TASK] * len(batch),
    }


def loader(examples, training):
    """Return the data loader of a list of examples; training examples are augmented."""
    return torch.utils.data.DataLoader(
        Clips(examples, augment=training),
        batch_size=BATCH,
        shuffle=training,
        drop_last=training,
        num_workers=WORKERS,
        collate_fn=collate,
        pin_memory=True,
    )


def loss_of(model, batch):
    """Return the loss of the model on one batch."""
    batch["raw_wav"] = batch["raw_wav"].cuda(non_blocking=True)
    batch["padding_mask"] = batch["padding_mask"].cuda(non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(batch, verbose=False)["loss"]


@torch.no_grad()
def validation_loss(model, batches):
    """Return the mean loss over the first 40 validation batches."""
    model.eval()
    losses = []
    for index, batch in enumerate(batches):
        if index >= VALIDATION_BATCHES:
            break
        losses.append(float(loss_of(model, batch)))
    model.train()
    return sum(losses) / len(losses)


def adapter_parameters(model):
    """Freeze everything but the LoRA adapters and return the adapter parameters."""
    adapters = []
    for name, parameter in model.named_parameters():
        is_adapter = "lora" in name.lower()
        parameter.requires_grad_(is_adapter)
        if is_adapter:
            adapters.append(parameter)
    return adapters


def train(size):
    """Fine-tune on one training size and write the model with the lowest validation loss."""
    torch.manual_seed(common.SEED)
    training, validation = common.split_validation(clips.examples(f"train_{size}"))
    print(f"training {len(training)}, validation {len(validation)}", flush=True)
    model = NatureLM.from_pretrained(MODEL_NAME).cuda()
    model.train()
    adapters = adapter_parameters(model)
    training_batches = loader(training, training=True)
    validation_batches = loader(validation, training=False)

    optimizer = torch.optim.AdamW(adapters, lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = int(len(training_batches) * EPOCHS / BATCHES_PER_STEP)
    validate_every = max(25, total_steps // VALIDATIONS)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=min(0.3, WARMUP_STEPS / total_steps),
        anneal_strategy="cos",
    )
    output_directory = config.DATA / "models" / "naturelm" / f"train_{size}"
    output_directory.mkdir(parents=True, exist_ok=True)

    step = 0
    best_loss = float("inf")
    validations_without_gain = 0
    finished = False
    for epoch in range(EPOCHS):
        if finished:
            break
        for batch_index, batch in enumerate(training_batches):
            loss = loss_of(model, batch)
            (loss / BATCHES_PER_STEP).backward()
            # Gradients are accumulated over eight batches, counted from the start of the
            # epoch; batches left over at its end count towards the first step of the next.
            if (batch_index + 1) % BATCHES_PER_STEP != 0:
                continue
            torch.nn.utils.clip_grad_norm_(adapters, 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(f"step {step}/{total_steps} loss {float(loss):.4f}", flush=True)
            if step % validate_every == 0:
                current_loss = validation_loss(model, validation_batches)
                print(f"step {step} validation loss {current_loss:.4f}", flush=True)
                if current_loss < best_loss:
                    best_loss = current_loss
                    validations_without_gain = 0
                    model.save_pretrained(str(output_directory))
                else:
                    validations_without_gain += 1
                    if validations_without_gain >= PATIENCE:
                        finished = True
                        break
            if step >= total_steps:
                finished = True
                break

    final_loss = validation_loss(model, validation_batches)
    print(f"final validation loss {final_loss:.4f}", flush=True)
    if final_loss < best_loss:
        model.save_pretrained(str(output_directory))
    print(output_directory)


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
