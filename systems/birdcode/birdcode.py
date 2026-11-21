"""Shared parts of the BirdCODE baseline: the released model, fine-tuned models, audio, clips."""
import json

import librosa
import numpy
import soundfile
import torch
from sound_event_detection.models import FrameDetector
from sound_event_detection.models.audioprotopnet_frame_detector import (
    _LinearLayerWithoutNegativeConnections as PrototypeClassifier,
)

from benchmark import config

PRETRAINED_REPOSITORY = "EarthSpeciesProject/sed-birdcode"

# Corpora sampled above this rate lie outside the 32-kHz input of BirdCODE.
HIGHEST_SAMPLING_RATE = 64000

# Length in seconds of the clip that holds one vocalization for identification.
CLIP_SECONDS = 5.0


def model_dir(model_name, size):
    """Return models/birdcode/<model_name>_<size> of the data root."""
    return config.DATA / "models" / "birdcode" / f"{model_name}_{size}"


def identification_clip_dir(size):
    """Return the folder of identification training clips of one training size."""
    return config.DATA / "birdcode" / f"identification_clips_{size}"


def load_pretrained():
    """Return the released BirdCODE model, downloaded from the Hugging Face Hub."""
    return FrameDetector.from_hf_hub(PRETRAINED_REPOSITORY)


def load_finetuned(directory):
    """Return a fine-tuned model in evaluation mode on the GPU.

    The prototypes and the classifier of the released model are replaced by empty ones of
    the fine-tuned size before the stored weights are loaded."""
    model = load_pretrained()
    taxa = (directory / "labels.txt").read_text().split("\n")[:-1]
    prototypes_per_class = model.classifier.features_per_output_class
    prototype_shape = model.encoder.prototype_vectors.shape[1:]
    empty_prototypes = torch.empty(prototypes_per_class * len(taxa), *prototype_shape)
    model.encoder.prototype_vectors = torch.nn.Parameter(empty_prototypes)
    model.classifier = PrototypeClassifier(prototypes_per_class * len(taxa), len(taxa))
    model.load_state_dict(torch.load(directory / "best_model.pt", map_location="cpu"))
    model.labels = taxa
    return model.eval().to("cuda")


def read_json(path):
    """Return the content of a JSON file."""
    with open(path) as handle:
        return json.load(handle)


def to_model_rate(waveform, sampling_rate, model):
    """Return a waveform as mono float32 at the sampling rate of the model."""
    if waveform.ndim > 1:
        waveform = waveform.mean(1)
    waveform = waveform.astype(numpy.float32)
    if sampling_rate != model.sample_rate:
        waveform = librosa.resample(
            waveform, orig_sr=sampling_rate, target_sr=model.sample_rate
        )
    return waveform.astype(numpy.float32)


def load_audio(path, model):
    """Read an audio file as mono float32 at the sampling rate of the model."""
    waveform, sampling_rate = soundfile.read(path, dtype="float32")
    return to_model_rate(waveform, sampling_rate, model)


def centre_vocalization(waveform, sampling_rate, onset, offset):
    """Cut one vocalization from its window and centre it in a clip of silence.

    The clip lasts CLIP_SECONDS, or as long as the vocalization if that is longer, at the
    sampling rate of the window. Returns (clip, onset, offset) with the position of the
    vocalization in the clip in seconds."""
    if waveform.ndim > 1:
        waveform = waveform.mean(1)
    first_sample = max(0, int(onset * sampling_rate))
    end_sample = min(len(waveform), int(offset * sampling_rate))
    if end_sample <= first_sample:
        end_sample = min(len(waveform), first_sample + 1)
    vocalization = waveform[first_sample:end_sample].astype(numpy.float32)
    clip_samples = max(int(round(CLIP_SECONDS * sampling_rate)), len(vocalization))
    samples_before = (clip_samples - len(vocalization)) // 2
    clip = numpy.zeros(clip_samples, numpy.float32)
    clip[samples_before:samples_before + len(vocalization)] = vocalization
    clip_onset = samples_before / sampling_rate
    clip_offset = (samples_before + len(vocalization)) / sampling_rate
    return clip, clip_onset, clip_offset
