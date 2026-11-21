"""Train TweetyNet on the joined training files of one training size.

    python -m systems.tweetynet.train --data DIR --size 50

One model for all corpora on spectrograms of the audio at 32 kHz, two classes per frame
(noise, vocal). A tenth of the training files of every corpus is held out; the weights of the
epoch with the highest frame F1 on these files are kept, and training stops when this F1 has
not improved for 8 epochs. Writes models/tweetynet/<size>/best.pt.
"""
import argparse
import time

import numpy as np
import torch

from benchmark import config
from systems.das import data
from systems.tweetynet import features

EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEED = 42
HIGHEST_CLASS_WEIGHT = 50.0


def frame_labels(file, frame_count, frame_seconds):
    """Return 1 for every frame that an annotation of the file touches, 0 elsewhere."""
    labels = np.zeros(frame_count, dtype=np.int64)
    for onset, offset in zip(file["onsets"], file["offsets"]):
        start = max(0, int(float(onset) / frame_seconds))
        end = min(frame_count, int(np.ceil(float(offset) / frame_seconds)))
        if end > start:
            labels[start:end] = 1
    return labels


def training_windows(files):
    """Return the spectrograms and frame labels of the files cut into windows of 176 frames.

    The windows do not overlap; frames after the last complete window of a file are left out.
    Shapes: (windows, 1, 128, 176) and (windows, 176)."""
    inputs = []
    targets = []
    for file in files:
        log_energy, frame_seconds = features.spectrogram(data.training_waveform(file))
        frame_count = log_energy.shape[1]
        labels = frame_labels(file, frame_count, frame_seconds)
        last_start = frame_count - features.WINDOW_FRAMES
        for start in range(0, last_start + 1, features.WINDOW_FRAMES):
            end = start + features.WINDOW_FRAMES
            inputs.append(log_energy[:, start:end])
            targets.append(labels[start:end])
    return np.stack(inputs)[:, None], np.stack(targets)


def class_weighted_loss(targets, device):
    """Return the cross-entropy loss with the vocal class weighted by noise frames per vocal
    frame of the training set, at most 50."""
    vocal_fraction = max(float(targets.mean()), 1e-4)
    weights = torch.tensor([1.0, (1 - vocal_fraction) / vocal_fraction], device=device)
    return torch.nn.CrossEntropyLoss(weight=weights.clamp(max=HIGHEST_CLASS_WEIGHT))


def train_epoch(model, optimizer, loss_function, inputs, targets, device):
    """Train for one pass over the shuffled windows and return the mean loss."""
    model.train()
    order = torch.randperm(len(inputs))
    loss_sum = 0.0
    for start in range(0, len(order), BATCH_SIZE):
        indices = order[start:start + BATCH_SIZE]
        loss = loss_function(model(inputs[indices].to(device)), targets[indices].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss) * len(indices)
    return loss_sum / len(inputs)


def validation_f1(model, inputs, targets, device):
    """Return the F1 of the vocal class over all frames of the validation windows."""
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    with torch.no_grad():
        for start in range(0, len(inputs), BATCH_SIZE):
            batch = torch.from_numpy(inputs[start:start + BATCH_SIZE]).to(device)
            predicted = model(batch).argmax(1).cpu().numpy()
            target = targets[start:start + BATCH_SIZE]
            true_positives += int(((predicted == 1) & (target == 1)).sum())
            false_positives += int(((predicted == 1) & (target == 0)).sum())
            false_negatives += int(((predicted == 0) & (target == 1)).sum())
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def copy_of_weights(model):
    """Return a copy of the weights of the model."""
    weights = {}
    for name, value in model.state_dict().items():
        weights[name] = value.detach().clone()
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = data.training_files(arguments.size)
    training_files, validation_files = data.split_training_files(files)
    training_inputs, training_targets = training_windows(training_files)
    validation_inputs, validation_targets = training_windows(validation_files)
    print(f"windows: {len(training_inputs)} training, {len(validation_inputs)} validation",
          flush=True)

    start_time = time.time()
    torch.manual_seed(SEED)
    model = features.build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = class_weighted_loss(training_targets, device)
    training_inputs = torch.from_numpy(training_inputs)
    training_targets = torch.from_numpy(training_targets)

    best_f1 = -1.0
    best_epoch = 0
    best_weights = None
    for epoch in range(EPOCHS):
        loss = train_epoch(
            model, optimizer, loss_function, training_inputs, training_targets, device
        )
        f1 = validation_f1(model, validation_inputs, validation_targets, device)
        print(f"epoch {epoch}: loss {loss:.4f}, validation frame F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_weights = copy_of_weights(model)
        elif epoch - best_epoch >= PATIENCE:
            break
    print(f"best validation frame F1 {best_f1:.4f} at epoch {best_epoch}, "
          f"{(time.time() - start_time) / 60:.1f} min")

    path = features.model_file(arguments.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, path)


if __name__ == "__main__":
    main()
