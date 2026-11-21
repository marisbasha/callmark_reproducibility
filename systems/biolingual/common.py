"""Taxa, validation split, augmentation and the classifier shared by the fine-tuning scripts."""
import random
from collections import defaultdict

import numpy as np
import torch

from benchmark import config, data
from systems.frozen_encoders import clips

MODEL_NAME = "davidrrobinson/BioLingual"
SAMPLING_RATE = 48000
EMBEDDING_SIZE = 512
VALIDATION_FRACTION = 0.1
SEED = 42


def taxa():
    """Return the sorted taxa of the dataset, the classes of the classifier."""
    names = set()
    for split in ("train_500", "test"):
        for row in data.table("classification", split, ("species",)):
            names.add(row["species"])
    return sorted(names)


def validation_taxa():
    """Return the taxa that enter the macro-F1 used to select the checkpoint.

    These are the taxa that the dataset marks as scorable."""
    names = set()
    for row in data.table("classification", "test", ("species", "scorable")):
        if row["scorable"]:
            names.add(row["species"])
    return names


def split_validation(examples):
    """Return (training examples, validation examples) of a training set.

    For every taxon with at least two recordings, a tenth of its recordings (at least one,
    never all) is held out with all their examples."""
    generator = random.Random(SEED)
    by_taxon = defaultdict(lambda: defaultdict(list))
    for example in examples:
        by_taxon[example["species"]][example["rec_id"]].append(example)
    training = []
    validation = []
    for by_recording in by_taxon.values():
        recordings = sorted(by_recording)
        generator.shuffle(recordings)
        if len(recordings) < 2:
            held_out = 0
        else:
            held_out = int(round(VALIDATION_FRACTION * len(recordings)))
            held_out = min(max(1, held_out), len(recordings) - 1)
        for recording in recordings[:held_out]:
            validation.extend(by_recording[recording])
        for recording in recordings[held_out:]:
            training.extend(by_recording[recording])
    return training, validation


def augmented(samples, generator):
    """Return a training clip with random gain, added noise, circular shift and polarity.

    Gain within 6 dB, Gaussian noise 20 to 40 dB below the clip, a shift of up to a tenth of
    the clip, and a sign flip with probability one half."""
    samples = samples * float(10 ** (generator.uniform(-6, 6) / 20))
    signal_to_noise = generator.uniform(20, 40)
    level = float(np.sqrt(np.mean(samples ** 2)))
    if level > 0:
        noise = generator.normal(0, level / (10 ** (signal_to_noise / 20)), len(samples))
        samples = samples + noise.astype(np.float32)
    if len(samples) > 8:
        samples = np.roll(samples, int(generator.uniform(-0.1, 0.1) * len(samples)))
    if generator.random() < 0.5:
        samples = -samples
    return np.clip(samples, -1.0, 1.0).astype(np.float32)


def item_generator(index):
    """Return the random generator of the training example at position `index`.

    It depends on the position only, so an example is augmented the same way in every epoch."""
    return np.random.default_rng(SEED * 1_000_003 + index)


class Clips(torch.utils.data.Dataset):
    """Vocalizations at 48 kHz with the index of their taxon."""

    def __init__(self, examples, taxon_names, augment):
        self.examples = examples
        self.class_of = {name: index for index, name in enumerate(taxon_names)}
        self.augment = augment

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        samples = clips.vocalization(example, SAMPLING_RATE)
        if self.augment:
            samples = augmented(samples, item_generator(index))
        return samples, self.class_of[example["species"]]


class Collate:
    """Turn a list of (clip, class) into the model input and a tensor of classes."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        waveforms = [samples for samples, _ in batch]
        classes = torch.tensor([taxon for _, taxon in batch], dtype=torch.long)
        inputs = self.processor(
            audios=waveforms,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        return inputs, classes


class Classifier(torch.nn.Module):
    """The audio encoder of BioLingual followed by a linear layer over all taxa."""

    def __init__(self, clap, class_count):
        super().__init__()
        self.clap = clap
        self.head = torch.nn.Linear(EMBEDDING_SIZE, class_count)

    def forward(self, inputs):
        return self.head(self.clap.get_audio_features(**inputs))


def limit_worker_threads(worker):
    """Keep every data-loading process on one thread."""
    torch.set_num_threads(1)


def model_file(size):
    """Return the path of the fine-tuned model of one training size."""
    return config.DATA / "models" / "biolingual" / f"train_{size}.pt"
