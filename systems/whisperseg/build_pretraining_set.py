"""Build the synthetic detection-pretraining set of WhisperSeg.

    python -m systems.whisperseg.build_pretraining_set [--data DIR] [--workers 16]

Writes prepared/whisperseg_pretraining/{train,val}/<name>.wav and .json (200 h and 3 h) and the
500-cap training chunks they are made from, prepared/whisperseg_pretraining/chunks/.

Every file is a background of 30 to 60 s with xeno-canto vocalizations inserted into it.

Backgrounds: stretches of at least 0.75 s without any annotation in the 500-cap training chunks
(labels/train_500_source_annotations.json adds the annotations of the original corpora that the
dataset does not contain), 20 ms away from the adjacent annotations. Chunks sampled above 48 kHz
are not used. Each file belongs to one corpus and sampling rate; its background pieces are joined
with 25-ms crossfades and it carries the spectrogram settings of that corpus. The hours of a
group are proportional to the square root of its background hours, at most 25 times those hours.

Vocalizations: event detections of BirdSet XCL (birdset_xcl/XCL/XCL_metadata.parquet) in the
recordings found in birdset_xcl/audio, except recordings whose xeno-canto identifier occurs in a
recording identifier of the benchmark and events of the noise cluster. 2 % of the recordings are
held out for val/. The boundaries of an event are trimmed to the frames within 20 dB of its peak
in-band energy; an event is dropped when the 30 ms outside its boundaries reach within 6 dB of
that peak, or when it is not 0.02 to 5 s long. The noise floor of the source recording is removed
with a spectral gate. The event is added at a signal-to-background ratio of -5 to 15 dB; 20 % of
the events overlap an earlier event of another taxon, at -8 to 8 dB.
"""
import argparse
import hashlib
import json
import math
import operator
import os
import random
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas
import soundfile
import soxr
from scipy.ndimage import uniform_filter
from scipy.signal import istft, stft

from benchmark import config, data
from systems.data import chunks
from systems.whisperseg import common

SEED = 20261030
HOURS = {"train": 200.0, "val": 3.0}

XCL_SAMPLE_RATE = 32000
MIN_STRETCH = 0.75              # shortest vocalization-free stretch used as background (s)
GUARD = 0.02                    # distance of a background stretch from an annotation (s)
MAX_BACKGROUND_RATE = 48000     # chunks sampled above this rate are not used (Hz)
MAX_REUSE = 25.0                # hours of a group per hour of its background
FILE_DURATION = (30.0, 60.0)
EVENT_RATE = (0.05, 0.45)       # inserted events per second, drawn per file
OVERLAP_PROBABILITY = 0.2
RATIO_DB = (-5.0, 15.0)         # signal-to-background ratio of an event
OVERLAP_RATIO_DB = (-8.0, 8.0)  # the same for an event placed over another event
MARGIN = 0.010                  # audio kept on each side of an event (s)
FADE = 0.005                    # raised-cosine fade inside the margin (s)
CROSSFADE = 0.025               # between background pieces (s)
MIN_SEPARATION = 0.030          # between events that do not overlap (s)
EVENT_DURATION = (0.02, 5.0)
VALIDATION_PERCENT = 2
MAX_EVENTS_PER_RECORDING = 8
POOL_FACTOR = 1.35              # events selected per event needed (some fail the refinement)
GATE_FFT = 512
GATE_HOP = 128
GATE_OVERSUBTRACTION = 2.0
GATE_CONTEXT_MS = 100           # audio gated on each side of the cut

# Background groups and event pools, set in main() and inherited by the worker processes.
SHARED = {}


def output_dir():
    """Return the folder of the pretraining set."""
    return common.pretraining_set_dir()


def xcl_dir():
    """Return the folder of the BirdSet XCL metadata and audio."""
    return config.DATA / "birdset_xcl"


# ---------------------------------------------------------------- backgrounds
def write_chunks(dataset_file):
    """Write the chunks of one dataset file as 16-bit WAV files."""
    directory = output_dir() / "chunks"
    for row in chunks.rows_with_audio(dataset_file):
        waveform, sample_rate = chunks.decode_audio(row)
        path = directory / wav_name(row["chunk_id"])
        soundfile.write(path, waveform, sample_rate, subtype="PCM_16")


def wav_name(chunk_id):
    """Return the file name of a chunk."""
    return f"{chunk_id}.wav"


def chunk_annotations(row, duration):
    """Return the onsets and offsets of a chunk, clipped to its audio."""
    onsets = []
    offsets = []
    for onset, offset in zip(row["onsets"], row["offsets"]):
        onset = max(0.0, float(onset))
        offset = min(duration, float(offset))
        if offset > onset:
            onsets.append(round(onset, 6))
            offsets.append(round(offset, 6))
    return onsets, offsets


def free_stretches(onsets, offsets, duration):
    """Return the (start, end) stretches of a chunk without annotations."""
    stretches = []
    covered_until = 0.0
    first = True
    for onset, offset in sorted(zip(onsets, offsets)):
        if first:
            start = covered_until
        else:
            start = covered_until + GUARD
        end = onset - GUARD
        if end - start >= MIN_STRETCH:
            stretches.append((start, end))
        covered_until = max(covered_until, offset)
        first = False
    if first:
        start = covered_until
    else:
        start = covered_until + GUARD
    if duration - start >= MIN_STRETCH:
        stretches.append((start, duration))
    return stretches


def load_backgrounds():
    """Return {(corpus, sampling rate): group} with the background stretches of every group.

    A group holds "settings" (spectrogram settings of its files), "stretches" [(wav path, start,
    end)] and "cumulative_seconds" (running total of the stretch lengths)."""
    source_annotations = data.read_json(config.label_file("train_500_source_annotations.json"))
    rows = data.table("segmentation", "train_500", ("chunk_id", "onsets", "offsets"))
    row_of_chunk = {}
    for row in rows:
        row_of_chunk[row["chunk_id"]] = row
    groups = {}
    # Chunks are taken in the order of their file names.
    for chunk_id in sorted(row_of_chunk, key=wav_name):
        row = row_of_chunk[chunk_id]
        wav_path = output_dir() / "chunks" / wav_name(chunk_id)
        info = soundfile.info(str(wav_path))
        if info.samplerate > MAX_BACKGROUND_RATE:
            continue
        duration = info.frames / info.samplerate
        corpus = row["dataset"]
        key = (corpus, info.samplerate)
        if key not in groups:
            settings = chunks.spectrogram_parameters(corpus, info.samplerate)
            groups[key] = {"key": key, "settings": settings, "stretches": []}
        onsets, offsets = chunk_annotations(row, duration)
        for onset, offset in source_annotations[chunk_id]:
            if onset < duration and offset > 0:
                onsets.append(max(0.0, onset))
                offsets.append(min(duration, offset))
        for start, end in free_stretches(onsets, offsets, duration):
            groups[key]["stretches"].append((str(wav_path), round(start, 4), round(end, 4)))
    for group in groups.values():
        lengths = []
        for _, start, end in group["stretches"]:
            lengths.append(end - start)
        group["cumulative_seconds"] = np.cumsum(lengths)
    return groups


def allocate_hours(groups, hours):
    """Return {group key: hours}: proportional to the square root of the background hours of a
    group, at most MAX_REUSE times those hours."""
    remaining = {}
    for key, group in groups.items():
        total = 0
        for _, start, end in group["stretches"]:
            total += end - start
        remaining[key] = total / 3600
    allocated = {}
    hours_left = hours
    while remaining:
        root_sum = 0
        for background_hours in remaining.values():
            root_sum += math.sqrt(background_hours)
        capped = False
        for key, background_hours in list(remaining.items()):
            share = hours_left * math.sqrt(background_hours) / root_sum
            if share > MAX_REUSE * background_hours:
                allocated[key] = MAX_REUSE * background_hours
                hours_left -= allocated[key]
                del remaining[key]
                capped = True
        if not capped:
            for key, background_hours in remaining.items():
                allocated[key] = hours_left * math.sqrt(background_hours) / root_sum
            break
    return allocated


def plan_files(groups, split, rng):
    """Return the files of one split: dictionaries with group key, duration, number of events
    and random seed."""
    allocated = allocate_hours(groups, HOURS[split])
    plans = []
    for key in sorted(groups):
        planned_seconds = 0.0
        while True:
            seconds_left = allocated[key] * 3600 - planned_seconds
            if seconds_left < 15:
                break
            if seconds_left < 30:
                duration = seconds_left
            else:
                duration = min(rng.uniform(*FILE_DURATION), seconds_left)
            num_events = max(1, int(round(duration * rng.uniform(*EVENT_RATE))))
            plans.append({
                "split": split,
                "index": len(plans),
                "key": key,
                "duration": duration,
                "num_events": num_events,
                "seed": rng.randrange(2 ** 31),
            })
            planned_seconds += duration
    return plans


# ---------------------------------------------------------------- xeno-canto events
def is_validation_recording(xeno_canto_id):
    """Return whether a xeno-canto recording is held out for the validation files."""
    digest = hashlib.md5(str(xeno_canto_id).encode()).hexdigest()
    return int(digest, 16) % 100 < VALIDATION_PERCENT


def benchmark_recording_numbers():
    """Return every number of 5 to 7 digits in a recording identifier of the benchmark."""
    numbers = set()
    for task in ("classification", "segmentation"):
        for split in ("train_500", "test"):
            for row in data.table(task, split, ("rec_id",)):
                for match in re.finditer(r"(?:XC)?#?(\d{5,7})", row["rec_id"]):
                    numbers.add(int(match.group(1)))
    return numbers


def usable_events(events, clusters, rng):
    """Return up to MAX_EVENTS_PER_RECORDING random events of a recording that are outside the
    noise cluster and of permitted duration."""
    kept = []
    for event, cluster in zip(events, clusters):
        duration = event[1] - event[0]
        if cluster != -1 and EVENT_DURATION[0] <= duration <= EVENT_DURATION[1]:
            kept.append((float(event[0]), float(event[1])))
    rng.shuffle(kept)
    return kept[:MAX_EVENTS_PER_RECORDING]


def select_events(events_needed):
    """Return {split: [(file name, xeno-canto id, taxon, [(start, end)])]}.

    Every taxon contributes the same number of events, taken from its recordings in turn."""
    columns = ["filepath", "ebird_code", "detected_events", "event_cluster"]
    metadata = pandas.read_parquet(xcl_dir() / "XCL" / "XCL_metadata.parquet", columns=columns)
    metadata["filepath"] = metadata.filepath.str.split("/").str[-1]
    local_files = set(os.listdir(xcl_dir() / "audio"))
    metadata = metadata[metadata.filepath.isin(local_files)].copy()
    metadata["xc"] = metadata.filepath.str.extract(r"XC(\d+)")[0].astype(int)
    metadata = metadata[~metadata.xc.isin(benchmark_recording_numbers())]
    held_out = metadata.xc.map(is_validation_recording)
    rng = random.Random(SEED)
    selected = {}
    for split in ("train", "val"):
        subset = metadata[held_out == (split == "val")]
        quota = math.ceil(events_needed[split] * POOL_FACTOR / subset.ebird_code.nunique())
        recordings_of_taxon = defaultdict(list)
        for row in subset.itertuples(index=False):
            events = usable_events(row.detected_events, row.event_cluster, rng)
            if events:
                recordings_of_taxon[row.ebird_code].append((row.filepath, int(row.xc), events))
        events_of_recording = defaultdict(list)
        for taxon in sorted(recordings_of_taxon):
            recordings = recordings_of_taxon[taxon]
            rng.shuffle(recordings)
            taken = 0
            turn = 0
            while taken < quota and any(recording[2] for recording in recordings):
                file_name, xeno_canto_id, events = recordings[turn % len(recordings)]
                turn += 1
                if events:
                    events_of_recording[(file_name, xeno_canto_id, taxon)].append(events.pop())
                    taken += 1
        selected[split] = []
        for (file_name, xeno_canto_id, taxon), events in events_of_recording.items():
            selected[split].append((file_name, xeno_canto_id, taxon, events))
    return selected


def refine_boundaries(audio, sample_rate, start, end):
    """Return the event boundaries trimmed to the frames within 20 dB of the peak in-band
    energy, or None if the event is cut off, silent or of unsuitable duration."""
    first_sample = max(0, int((start - 0.05) * sample_rate))
    last_sample = min(len(audio), int((end + 0.05) * sample_rate))
    segment = audio[first_sample:last_sample]
    if len(segment) < 512:
        return None
    frequencies, times, spectrum = stft(segment, sample_rate, nperseg=512, noverlap=384,
                                        boundary=None)
    in_band = (frequencies >= 1000) & (frequencies <= min(15000, sample_rate / 2))
    energy_db = 10 * np.log10((np.abs(spectrum[in_band]) ** 2).sum(0) + 1e-12)
    times = times + first_sample / sample_rate
    inside = (times >= start) & (times <= end)
    if inside.sum() < 3:
        return None
    peak = energy_db[inside].max()
    if peak < -100:
        return None
    before = (times >= start - 0.03) & (times < start)
    after = (times > end) & (times <= end + 0.03)
    adjacent = (~inside) & (before | after)
    if adjacent.any() and energy_db[adjacent].max() >= peak - 6:
        return None
    active = np.where(inside & (energy_db >= peak - 20))[0]
    hop = 128 / sample_rate
    onset = max(start, times[active[0]] - hop)
    offset = min(end, times[active[-1]] + hop)
    if not EVENT_DURATION[0] <= offset - onset <= EVENT_DURATION[1]:
        return None
    return onset, offset


def noise_spectrum(audio, sample_rate, rng):
    """Return the mean power spectrum of the quietest 20 % of the frames of a recording (of a
    random 60 s of a longer one)."""
    max_samples = 60 * sample_rate
    if len(audio) > max_samples:
        first_sample = rng.randrange(len(audio) - max_samples)
        audio = audio[first_sample:first_sample + max_samples]
    _, _, spectrum = stft(audio, sample_rate, nperseg=GATE_FFT, noverlap=GATE_FFT - GATE_HOP)
    power = np.abs(spectrum) ** 2
    frame_energy = power.sum(0)
    quiet = frame_energy <= np.percentile(frame_energy, 20)
    return power[:, quiet].mean(1)


def spectral_gate(segment, sample_rate, noise):
    """Remove the noise floor: gain max(0, 1 - 2 noise / power), with the power smoothed over
    3 frequency bins and 3 frames."""
    _, _, spectrum = stft(segment, sample_rate, nperseg=GATE_FFT, noverlap=GATE_FFT - GATE_HOP)
    power = uniform_filter(np.abs(spectrum) ** 2, size=3, mode="nearest")
    gain = np.clip(1 - GATE_OVERSUBTRACTION * noise[:, None] / np.maximum(power, 1e-20), 0, 1)
    _, gated = istft(spectrum * gain, sample_rate, nperseg=GATE_FFT,
                     noverlap=GATE_FFT - GATE_HOP)
    return gated[:len(segment)].astype(np.float32)


def extract_events(task):
    """Return the refined, noise-gated cuts of the selected events of one recording.

    A cut holds the event and MARGIN seconds on each side, as 16-bit samples of the peak-scaled
    waveform; "lead" is the time of the event onset within the cut."""
    file_name, xeno_canto_id, taxon, events = task
    audio, sample_rate = soundfile.read(str(xcl_dir() / "audio" / file_name), dtype="float32",
                                        always_2d=True)
    audio = audio.mean(1)
    assert sample_rate == XCL_SAMPLE_RATE, file_name
    noise = noise_spectrum(audio, sample_rate, random.Random(xeno_canto_id))
    context = GATE_CONTEXT_MS * sample_rate // 1000
    cuts = []
    for start, end in events:
        refined = refine_boundaries(audio, sample_rate, start, end)
        if refined is None:
            continue
        onset, offset = refined
        first_sample = max(0, int(round((onset - MARGIN) * sample_rate)))
        last_sample = min(len(audio), int(round((offset + MARGIN) * sample_rate)))
        context_start = max(0, first_sample - context)
        context_end = min(len(audio), last_sample + context)
        gated = spectral_gate(audio[context_start:context_end], sample_rate, noise)
        cut = gated[first_sample - context_start:last_sample - context_start]
        peak = float(np.abs(cut).max())
        if peak <= 0:
            continue
        cuts.append({
            "xc": xeno_canto_id,
            "taxon": taxon,
            "onset": round(onset, 4),
            "lead": onset - first_sample / sample_rate,
            "duration": offset - onset,
            "scale": peak,
            "samples": np.round(cut / peak * 32767).astype(np.int16),
        })
    return cuts


def event_order(event):
    """Sort key that makes the order of the extracted events independent of the processes."""
    return (event["taxon"], event["xc"], event["onset"])


# ---------------------------------------------------------------- mixtures
def root_mean_square(samples):
    """Return the root mean square of `samples`, 0 if empty."""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def make_background(rng, group, duration, sample_rate):
    """Return a background of `duration` seconds: random pieces of the stretches of the group
    (a stretch is chosen in proportion to its length), joined with crossfades."""
    stretches = group["stretches"]
    cumulative = group["cumulative_seconds"]
    crossfade_samples = int(CROSSFADE * sample_rate)
    target_samples = int(duration * sample_rate)
    pieces = []
    total_samples = 0
    while total_samples < target_samples:
        index = int(np.searchsorted(cumulative, rng.random() * cumulative[-1]))
        wav_path, start, end = stretches[min(index, len(stretches) - 1)]
        length = end - start
        seconds_left = (target_samples - total_samples) / sample_rate + 2 * CROSSFADE + 0.05
        if length > 20:
            piece_length = min(length, seconds_left, rng.uniform(5.0, 20.0))
        else:
            piece_length = min(length, seconds_left)
        piece_start = start + rng.uniform(0, max(0.0, length - piece_length))
        piece, _ = soundfile.read(wav_path, start=int(piece_start * sample_rate),
                                  stop=int((piece_start + piece_length) * sample_rate),
                                  dtype="float32")
        if len(piece) < 2 * crossfade_samples + 1:
            continue
        if pieces:
            fade = np.linspace(0, 1, crossfade_samples, dtype=np.float32)
            previous = pieces[-1]
            previous[-crossfade_samples:] = (previous[-crossfade_samples:] * (1 - fade)
                                             + piece[:crossfade_samples] * fade)
            piece = piece[crossfade_samples:]
        pieces.append(piece)
        total_samples += len(piece)
    return np.concatenate(pieces)[:target_samples]


def free_position(rng, occupied, duration, lowest, highest):
    """Return a random onset that keeps MIN_SEPARATION from the events placed so far, or None
    after 60 failed draws."""
    for _ in range(60):
        onset = rng.uniform(lowest, highest)
        is_free = True
        for other_onset, other_offset in occupied:
            if onset + duration + MIN_SEPARATION > other_onset \
                    and onset < other_offset + MIN_SEPARATION:
                is_free = False
        if is_free:
            return onset
    return None


def event_waveform(event, sample_rate):
    """Return the cut of an event at the sampling rate of the background, with faded ends."""
    waveform = event["samples"].astype(np.float32) / 32767 * event["scale"]
    if sample_rate != XCL_SAMPLE_RATE:
        waveform = soxr.resample(waveform, XCL_SAMPLE_RATE, sample_rate)
    fade_samples = int(FADE * sample_rate)
    if len(waveform) > 2 * fade_samples and fade_samples > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_samples))
        waveform[:fade_samples] *= ramp
        waveform[-fade_samples:] *= ramp[::-1]
    return waveform


def overlapping_onset(rng, placed, event, lowest, highest):
    """Return an onset at which the event overlaps an earlier event of another taxon, or None.

    With probability OVERLAP_PROBABILITY a partner is drawn; the event starts between 75 % of
    its duration before the partner's onset and 25 % of its duration before the partner's
    offset."""
    candidates = []
    for other in placed:
        if other["taxon"] != event["taxon"]:
            candidates.append(other)
    if not candidates or rng.random() >= OVERLAP_PROBABILITY:
        return None
    partner = rng.choice(candidates)
    duration = event["duration"]
    earliest = partner["onset"] - 0.75 * duration
    onset = rng.uniform(earliest, max(earliest, partner["offset"] - 0.25 * duration))
    onset = min(max(onset, lowest), highest)
    if onset < partner["offset"] and onset + duration > partner["onset"]:
        return onset
    return None


def build_file(plan):
    """Write the audio and the labels of one planned file."""
    rng = random.Random(plan["seed"])
    corpus, sample_rate = plan["key"]
    group = SHARED["groups"][plan["key"]]
    background = make_background(rng, group, plan["duration"], sample_rate)
    file_duration = len(background) / sample_rate
    mixture = background.astype(np.float64).copy()
    events = SHARED["events"][plan["split"]]
    occupied = []
    placed = []
    for event_index in plan["event_indices"]:
        event = events[event_index % len(events)]
        waveform = event_waveform(event, sample_rate)
        lead = event["lead"]
        duration = event["duration"]
        # Range of the onset that keeps the whole cut inside the file.
        lowest = 0.1 + lead
        highest = file_duration - 0.1 - len(waveform) / sample_rate + lead
        if highest <= lowest:
            continue
        onset = overlapping_onset(rng, placed, event, lowest, highest)
        overlaps = onset is not None
        if not overlaps:
            onset = free_position(rng, occupied, duration, lowest, highest)
            if onset is None:
                continue
        occupied.append((onset, onset + duration))
        first_sample = int(round((onset - lead) * sample_rate))
        waveform = waveform[:len(mixture) - first_sample]
        onset = first_sample / sample_rate + lead
        offset = onset + duration
        lead_samples = int(round(lead * sample_rate))
        event_samples = int(round(offset * sample_rate)) - int(round(onset * sample_rate))
        event_level = root_mean_square(waveform[lead_samples:lead_samples + event_samples])
        centre = 0.5 * (onset + offset)
        window_start = max(0, int((centre - 2.5) * sample_rate))
        window_end = min(len(background), int((centre + 2.5) * sample_rate))
        background_level = root_mean_square(background[window_start:window_end])
        if overlaps:
            ratio_db = rng.uniform(*OVERLAP_RATIO_DB)
        else:
            ratio_db = rng.uniform(*RATIO_DB)
        if event_level > 0 and background_level > 0:
            gain = background_level * 10 ** (ratio_db / 20) / event_level
        else:
            gain = 1.0
        mixture[first_sample:first_sample + len(waveform)] += gain * waveform
        placed.append({"onset": round(onset, 4), "offset": round(offset, 4),
                       "taxon": event["taxon"]})
    peak = float(np.abs(mixture).max())
    if peak > 0.99:
        mixture *= 0.99 / peak
    split = plan["split"]
    name = f"{corpus}__sr{sample_rate}__{split}{plan['index']:05d}"
    directory = output_dir() / split
    soundfile.write(directory / f"{name}.wav", mixture.astype(np.float32), sample_rate,
                    subtype="PCM_16")
    placed.sort(key=operator.itemgetter("onset"))
    labels = dict(group["settings"])
    labels["onset"] = [vocalization["onset"] for vocalization in placed]
    labels["offset"] = [min(vocalization["offset"], file_duration) for vocalization in placed]
    labels["cluster"] = ["Vocal"] * len(placed)
    labels["species"] = corpus
    (directory / f"{name}.json").write_text(json.dumps(labels))
    return len(placed)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="data root (default: the variable CALLMARK_DATA)")
    parser.add_argument("--workers", type=int, default=16, help="parallel processes")
    arguments = parser.parse_args()
    config.use_data_root(arguments.data)

    for name in ("chunks", "train", "val"):
        (output_dir() / name).mkdir(parents=True)
    with Pool(arguments.workers) as pool:
        pool.map(write_chunks, chunks.dataset_files("train_500"))

    groups = load_backgrounds()
    rng = random.Random(SEED)
    plans = {}
    events_needed = {}
    for split in ("train", "val"):
        plans[split] = plan_files(groups, split, rng)
        events_needed[split] = sum(plan["num_events"] for plan in plans[split])
        print(f"{split}: {len(plans[split])} files, {events_needed[split]} events", flush=True)

    selected = select_events(events_needed)
    events = {}
    for split in ("train", "val"):
        events[split] = []
        with Pool(arguments.workers) as pool:
            for cuts in pool.imap_unordered(extract_events, selected[split], chunksize=16):
                events[split] += cuts
        events[split].sort(key=event_order)
        random.Random(SEED).shuffle(events[split])
        print(f"{split}: {len(events[split])} events extracted", flush=True)
        # Each file takes the next events of the shuffled pool; a short pool is reused.
        next_event = 0
        for plan in plans[split]:
            plan["event_indices"] = range(next_event, next_event + plan["num_events"])
            next_event += plan["num_events"]

    SHARED["groups"] = groups
    SHARED["events"] = events
    with Pool(arguments.workers) as pool:
        placed = pool.map(build_file, plans["train"] + plans["val"], chunksize=4)
    print(f"{len(placed)} files with {sum(placed)} vocalizations in {output_dir()}")


if __name__ == "__main__":
    main()
