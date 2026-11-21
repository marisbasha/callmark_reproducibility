"""Train one of the three systems on recordings of pairs and select a checkpoint.

All systems start from the same WhisperSeg checkpoint and use the same optimizer, schedule
and number of steps. After every --validate-every steps the model is converted to
CTranslate2 float16, run on the validation set and scored; the checkpoint with the highest
validation F1 is kept, the earlier one on ties.

  microphone     input: the microphone. Target: the vocalizations of both birds.
  accelerometer  input: the accelerometer of the focal bird. Target: its vocalizations.
  multichannel   input: focal accelerometer, accelerometer of the other bird, microphone.
                 Target: the vocalizations of the focal bird.

Writes models/<system>/ct2 (the selected model), models/<system>/checkpoint and
models/<system>/validation.json.
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from data import SYSTEMS, data_root
from score import greedy_matches, maximum_matches

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mcwhisperseg"))
import datautils  # noqa: E402
from model import load_model, save_model  # noqa: E402

SPEC_COLUMNS = 1000
TARGET_TOKENS = {"microphone": 128, "accelerometer": 100, "multichannel": 100}
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
CROSSTALK_PROBABILITY = 0.5
CROSSTALK_GAIN = (0.1, 0.5)
OTHER_ONLY_WEIGHT = 2.0


def attach_other_bird(labels, label_paths):
    """Add the vocalizations of the other bird of the pair to the label of every focal
    example. The labels are changed in place."""
    for label, path in zip(labels, label_paths):
        path = Path(path)
        match = re.search(r"_channel_([12])_as_target_channel_1.json$", path.name)
        chunk = path.name[:match.start()]
        other_bird = 3 - int(match[1])
        other_path = path.with_name(f"{chunk}_channel_{other_bird}_as_target_channel_1.json")
        reference = json.loads(other_path.read_text())
        label["other_onset"] = np.asarray(reference["onset"], dtype=float)
        label["other_offset"] = np.asarray(reference["offset"], dtype=float)


def slice_with_other_bird(audios, labels):
    """Audio clips and their labels, cut like datautils.slice_audios_and_labels, with the
    other bird's vocalizations carried along."""
    sliced_audios = []
    sliced_labels = []
    for audio, label in zip(audios, labels):
        clips, clip_labels = datautils.slice_audio_and_label(audio, label, SPEC_COLUMNS)
        clip_duration = SPEC_COLUMNS * label["spec_time_step"]
        # slice_audio_and_label puts one clip duration of silence in front of the audio.
        onset = label["other_onset"] + clip_duration
        offset = label["other_offset"] + clip_duration
        for i, (clip, clip_label) in enumerate(zip(clips, clip_labels)):
            start = i * clip_duration
            end = start + clip.shape[1] / label["sr"]
            keep = (onset < end) & (offset > start)
            clip_label["other_onset"] = np.maximum(onset[keep], start) - start
            clip_label["other_offset"] = np.minimum(offset[keep], end) - start
        sliced_audios.extend(clips)
        sliced_labels.extend(clip_labels)
    return sliced_audios, sliced_labels


class FocalDataset(datautils.VocalSegDataset):
    """Training windows of (focal accelerometer, other accelerometer, microphone).

    In half of the windows the other bird's accelerometer waveform is added to the focal
    channel with a random gain. The target stays the focal bird's vocalizations, also where
    both birds vocalize. Windows in which only the other bird vocalizes are flagged."""

    def __init__(self, *args, focal_only):
        super().__init__(*args)
        self.focal_only = focal_only

    def __getitem__(self, index):
        """One training window that starts at a random sample of the clip."""
        audio = self.audio_list[index]
        label = self.label_list[index]
        rate = label["sr"]
        time_step = label["spec_time_step"]
        extractor = self.feature_extractor_bank[f"{rate}-{time_step}-{label['min_frequency']}"]
        n_samples = int(np.round(self.total_spec_columns * time_step * rate))
        n_possible_starts = min(n_samples + 1, audio.shape[1] - extractor.n_fft + 1)
        first = np.random.choice(n_possible_starts)
        clip = audio[:, first:first + n_samples].copy()
        start = first / rate
        end = (first + clip.shape[1]) / rate
        own = (label["onset"] < end) & (label["offset"] > start)
        other = (label["other_onset"] < end) & (label["other_offset"] > start)
        if np.random.rand() < CROSSTALK_PROBABILITY:
            gain = np.float32(np.random.uniform(*CROSSTALK_GAIN))
            clip[0] = clip[0] + gain * clip[1]
        onsets = np.maximum(label["onset"][own], start) - start
        offsets = np.minimum(label["offset"][own], end) - start
        text = self.species_codebook.get(label["species"], "<|unknown|>")
        for onset, offset, cluster in zip(onsets, offsets, label["cluster_id"][own]):
            onset_column = self.map_time_to_spec_col_index(onset, time_step)
            offset_column = self.map_time_to_spec_col_index(offset, time_step)
            text += "<|%d|>%d<|%d|>" % (onset_column, cluster, offset_column)
        padding = n_samples - clip.shape[1]
        clip = np.pad(clip, ((0, 0), (0, padding))).astype(np.float32)
        channels = clip[:1] if self.focal_only else clip
        features = []
        for channel in channels:
            extracted = extractor(channel, sampling_rate=rate, padding="do_not_pad")
            features.append(extracted["input_features"][0])
        tokens = self.tokenizer.encode(
            text, max_length=self.max_length + 1, truncation=True, padding=True)
        target = tokens[1:]
        decoder_input = tokens[:-1]
        target += [-100] * (self.max_length - len(target))
        decoder_input += [self.tokenizer.pad_token_id] * (self.max_length - len(decoder_input))
        return {
            "input_features": np.concatenate(features),
            "decoder_input_ids": np.asarray(decoder_input),
            "labels": np.asarray(target),
            "other_only": np.bool_(other.any() and not own.any()),
        }


def loss_function(logits, labels, other_only=None):
    """Cross-entropy of the target tokens of a batch. other_only flags the examples of
    the focal systems in which only the other bird vocalizes."""
    per_token = F.cross_entropy(
        logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    valid = labels.ne(-100)
    if other_only is None:
        # microphone: mean over the tokens of each example, then over examples
        return (per_token.sum(1) / valid.sum(1)).mean()
    # focal systems: pooled over all target tokens of the batch, with a higher weight on the
    # tokens of examples in which only the other bird vocalizes
    weights = torch.where(other_only, OTHER_ONLY_WEIGHT, 1.0).to(per_token)[:, None]
    return (per_token * weights).sum() / (valid * weights).sum()


def validation_score(root, system, prediction_folder):
    """Pooled F1 over the validation examples, both boundaries within 20 ms."""
    reference_folder = root / ("val_microphone" if system == "microphone" else "val")
    tp = 0
    n_pred = 0
    n_ref = 0
    for path in sorted(prediction_folder.glob("*.json")):
        prediction = json.loads(path.read_text())["predictions"][0]
        predicted = list(zip(prediction["onset"], prediction["offset"]))
        label = json.loads((reference_folder / path.name).read_text())
        reference = list(zip(label["onset"], label["offset"]))
        if system == "microphone":
            tp += int(maximum_matches(predicted, reference).sum())
        else:
            tp += greedy_matches(predicted, reference)
        n_pred += len(predicted)
        n_ref += len(reference)
    f1 = 2 * tp / max(n_pred + n_ref, 1)
    return {"f1": f1, "tp": tp, "n_pred": n_pred, "n_ref": n_ref}


def validate(root, system, model, tokenizer, step, folder):
    """Save the model as checkpoint-<step>, convert it to CTranslate2 float16 as ct2-<step>,
    run it on the validation set and return its step and validation score."""
    save_model(model, tokenizer, step, str(folder), -1)
    checkpoint = folder / f"checkpoint-{step}"
    ct2 = folder / f"ct2-{step}"
    predictions = root / "predictions" / system / f"val_step{step}"
    convert = [
        sys.executable, str(HERE / "mcwhisperseg/convert_hf_to_ct2.py"),
        "--model", str(checkpoint),
        "--output_dir", str(ct2),
        "--quantization", "float16",
    ]
    infer = [
        sys.executable, str(HERE / "infer.py"),
        "--system", system,
        "--set", "val",
        "--data", str(root.parent),
        "--model", str(ct2),
        "--out", str(predictions),
    ]
    subprocess.run(convert, check=True)
    subprocess.run(infer, check=True)
    return {"step": step, **validation_score(root, system, predictions)}


def remove_checkpoint(folder, step):
    """Delete the checkpoint and the converted model saved at this step."""
    shutil.rmtree(folder / f"checkpoint-{step}")
    shutil.rmtree(folder / f"ct2-{step}")


def seed_everything(seed):
    """Seed the random number generators of Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training_set(root, system, model, tokenizer):
    """Dataset of the training windows of a system. The codebook of the vocalization
    classes found in the labels is stored in the model configuration."""
    focal = system != "microphone"
    train_folder = root / ("train" if focal else "train_microphone")
    audio_paths, label_paths = datautils.get_audio_and_label_paths(str(train_folder))
    codebook = datautils.get_cluster_codebook(label_paths, {})
    model.config.cluster_codebook = codebook
    audios, labels = datautils.load_data(
        audio_paths, label_paths, cluster_codebook=codebook, n_threads=20)
    dataset_arguments = (
        tokenizer, TARGET_TOKENS[system], SPEC_COLUMNS, model.config.species_codebook)
    if focal:
        attach_other_bird(labels, label_paths)
        audios, labels = slice_with_other_bird(audios, labels)
        focal_only = system == "accelerometer"
        return FocalDataset(audios, labels, *dataset_arguments, focal_only=focal_only)
    audios, labels = datautils.slice_audios_and_labels(audios, labels, SPEC_COLUMNS)
    return datautils.VocalSegDataset(audios, labels, *dataset_arguments)


def without_weight_decay(parameter_name):
    """Whether a parameter is a bias or a layer normalization weight."""
    return "bias" in parameter_name or "LayerNorm.weight" in parameter_name


def make_optimizer(model, learning_rate):
    """AdamW with weight decay on all parameters except biases and layer normalization
    weights."""
    decayed = []
    not_decayed = []
    for name, parameter in model.named_parameters():
        if without_weight_decay(name):
            not_decayed.append(parameter)
        else:
            decayed.append(parameter)
    parameter_groups = [
        {"params": decayed, "weight_decay": WEIGHT_DECAY},
        {"params": not_decayed, "weight_decay": 0.0},
    ]
    return AdamW(parameter_groups, lr=learning_rate)


def training_step(model, batch, optimizer, scheduler, scaler):
    """Update the model on one batch in mixed precision and return the loss."""
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(
            input_features=batch["input_features"],
            decoder_input_ids=batch["decoder_input_ids"])
        loss = loss_function(output.logits, batch["labels"], batch.get("other_only"))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--data", help="data root (default: $CALLMARK_DATA)")
    parser.add_argument("--initial-model", default="nccratliri/whisperseg-animal-vad")
    parser.add_argument("--seed", type=int, default=66100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--steps", type=int, default=34185)
    parser.add_argument("--validate-every", type=int, default=6837)
    args = parser.parse_args()
    root = data_root(args.data)
    folder = root / "models" / args.system
    folder.mkdir(parents=True)

    seed_everything(args.seed)

    n_channels = 3 if args.system == "multichannel" else 1
    model, tokenizer = load_model(args.initial_model, SPEC_COLUMNS, num_channels=n_channels)
    model = model.cuda()
    dataset = training_set(root, args.system, model, tokenizer)

    epoch = 0

    def seed_worker(worker):
        # The loader starts new workers for every pass over the data; seeding them with
        # the pass number makes every pass draw different windows.
        np.random.seed(epoch + worker)
        torch.manual_seed(epoch + worker)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed))

    optimizer = make_optimizer(model, args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, args.steps)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    step = 0
    losses = []
    history = []
    best = None
    while step < args.steps:
        for batch in loader:
            for key in batch:
                batch[key] = batch[key].cuda()
            loss = training_step(model, batch, optimizer, scheduler, scaler)
            step += 1
            losses.append(loss)
            if step % 100 == 0:
                print(f"step {step}  loss {np.mean(losses):.4f}", flush=True)
                losses = []
            if step % args.validate_every == 0 or step == args.steps:
                model.eval()
                score = validate(root, args.system, model, tokenizer, step, folder)
                model.train()
                history.append(score)
                print("validation", json.dumps(score), flush=True)
                # Keep the checkpoint with the highest validation F1, the earlier one on ties.
                if best is None or score["f1"] > best["f1"]:
                    superseded = best
                    best = score
                else:
                    superseded = score
                if superseded is not None:
                    remove_checkpoint(folder, superseded["step"])
            if step == args.steps:
                break
        epoch += 1

    (folder / f"checkpoint-{best['step']}").rename(folder / "checkpoint")
    (folder / f"ct2-{best['step']}").rename(folder / "ct2")
    summary = {"history": history, "selected": best}
    (folder / "validation.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
