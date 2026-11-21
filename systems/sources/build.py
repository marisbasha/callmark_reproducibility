"""The rules that turn the original corpora into the benchmark: duration filter, chunks,
split into training and test recordings, and the choice of the identification examples.

1. A vocalization longer than the 95th percentile of the identified vocalizations of its corpus
   is removed.
2. Every recording is cut into chunks of about 60 s. A cut that falls inside a vocalization is
   moved to the nearer edge of that vocalization.
3. Whole recordings go to the test set until every taxon has about 20 % of its vocalizations
   there.
4. At most 500 vocalizations per taxon and side become identification examples, in a random
   order that defines the training sizes 50, 100 and 500.

Vocalizations without taxon stay in their chunks and are skipped by everything that counts
taxa.
"""
import hashlib
import random
from collections import defaultdict

import numpy

from systems.sources import corpora

CHUNK_SECONDS = 60.0
DURATION_PERCENTILE = 95
SEED = 42
TEST_FRACTION = 0.20
EXAMPLES_PER_TAXON = 500
SIZES = (50, 100, 500)

# A taxon is scored if it occurs in more than one recording and has at least this many
# identification examples.
MIN_TRAIN_EXAMPLES = 25
MIN_TEST_EXAMPLES = 10

# Name given to unidentified vocalizations of the Sierra Nevada corpus. It is treated as a
# taxon in the split and the caps, has no identification examples and is not scored.
UNIDENTIFIED = "unknown"

# Corpora with one taxon: their recordings are assigned largest first.
SINGLE_TAXON_CORPORA = (
    "budgie",
    "canary",
    "infant_marmosets",
    "ozf",
    "right_whale",
    "singing_mice",
    "spiders",
    "tworadiobirds",
    "wytham",
)


# ---------------------------------------------------------------- chunks
def event_order(event):
    """Sort key of an event: onset, offset, taxon (no taxon sorts first)."""
    return (event[0], event[1], event[2] or "")


def cut_chunks(events):
    """Cut the span of a recording up to its last offset into chunks.

    Returns [(start, end, events)] for the chunks that hold an onset, with event times relative
    to the chunk start. A chunk ends at the last offset of its events if that is later than the
    cut."""
    events = sorted(events, key=event_order)
    span = max(offset for _, offset, _ in events)

    boundaries = [0.0]
    current = 0.0
    while current + CHUNK_SECONDS < span - 1e-6:
        boundary = current + CHUNK_SECONDS
        for onset, offset, _ in events:
            if onset < boundary < offset:
                if boundary - onset <= offset - boundary:
                    boundary = onset
                else:
                    boundary = offset
                break
        if boundary <= current + 1e-6:
            boundary = current + CHUNK_SECONDS
        boundaries.append(boundary)
        current = boundary
    boundaries.append(span)
    boundaries = sorted(set(boundaries))

    chunks = []
    for start, end in zip(boundaries, boundaries[1:]):
        inside = []
        for onset, offset, taxon in events:
            if start <= onset < end:
                inside.append((onset, offset, taxon))
        if inside:
            last_offset = max(offset for _, offset, _ in inside)
            relative = []
            for onset, offset, taxon in inside:
                relative.append((onset - start, offset - start, taxon))
            chunks.append((start, max(end, last_offset), relative))
    return chunks


def longest_durations(recordings):
    """Return {corpus: longest duration kept}, the percentile of its identified vocalizations."""
    durations = defaultdict(list)
    for item in recordings:
        for onset, offset, taxon in item["events"]:
            if taxon is not None:
                durations[item["corpus"]].append(offset - onset)
    longest = {}
    for corpus, values in durations.items():
        longest[corpus] = float(numpy.percentile(values, DURATION_PERCENTILE))
    return longest


def build_chunks(recordings):
    """Return (chunks, longest durations) of all recordings.

    A chunk is a dictionary with chunk_id, rec_id, corpus, audio, chunk_start, chunk_end
    (seconds in the recording) and events [{"onset", "offset", "taxon"}] relative to the chunk
    start. Times are rounded to 0.1 ms."""
    longest = longest_durations(recordings)
    chunks = []
    for item in recordings:
        kept = []
        for onset, offset, taxon in item["events"]:
            if 0 < offset - onset <= longest[item["corpus"]]:
                kept.append((onset, offset, taxon))
        if not kept:
            continue
        for index, (start, end, inside) in enumerate(cut_chunks(kept)):
            events = []
            for onset, offset, taxon in inside:
                events.append({"onset": round(onset, 4), "offset": round(offset, 4),
                               "taxon": taxon})
            # The end and the offsets are rounded separately; the chunk must still contain
            # every event.
            last_offset = max(event["offset"] for event in events)
            chunk_end = max(round(end, 4), round(start, 4) + last_offset)
            chunks.append({
                "chunk_id": f"{item['rec_id']}__c{index:03d}",
                "rec_id": item["rec_id"],
                "corpus": item["corpus"],
                "audio": item["audio"],
                "chunk_start": round(start, 4),
                "chunk_end": chunk_end,
                "events": events,
            })
    return chunks, longest


# ---------------------------------------------------------------- split
def tie_break(rec_id):
    """Return a fixed pseudo-random number in [0, 1) for a recording."""
    digest = hashlib.blake2b(f"{SEED}:{rec_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2 ** 64


def split_recordings(chunks):
    """Return {rec_id: "train" or "test"}.

    Recordings of the single-taxon corpora are taken largest first into the test set while they
    fit into the remaining 20 % of the taxon (with 15 % slack). The other recordings, those with
    most taxa first, go to the test set when they fill more of the missing test vocalizations of
    their taxa than they exceed them."""
    counts = defaultdict(lambda: defaultdict(int))
    corpus_of = {}
    total = defaultdict(int)
    for chunk in chunks:
        corpus_of.setdefault(chunk["rec_id"], chunk["corpus"])
        recording_counts = counts[chunk["rec_id"]]
        for event in chunk["events"]:
            if event["taxon"] is not None:
                recording_counts[event["taxon"]] += 1
                total[event["taxon"]] += 1
    size = {}
    for rec_id, recording_counts in counts.items():
        size[rec_id] = sum(recording_counts.values())
    target = {}
    for taxon, number in total.items():
        target[taxon] = TEST_FRACTION * number

    side = {}
    in_test = defaultdict(int)
    for corpus in SINGLE_TAXON_CORPORA:
        rec_ids = [rec_id for rec_id in counts if corpus_of[rec_id] == corpus]
        if not rec_ids:
            continue
        taxon = next(iter(counts[rec_ids[0]]))
        budget = target[taxon]
        largest_first = sorted(rec_ids, key=lambda rec_id: (-size[rec_id], tie_break(rec_id)))
        for rec_id in largest_first:
            if budget > 0 and size[rec_id] <= budget * 1.15:
                side[rec_id] = "test"
                in_test[taxon] += size[rec_id]
                budget -= size[rec_id]
            else:
                side[rec_id] = "train"
        if in_test[taxon] == 0 and len(rec_ids) > 1:
            smallest = min(rec_ids, key=size.get)
            side[smallest] = "test"
            in_test[taxon] += size[smallest]

    missing = {}
    for taxon in target:
        missing[taxon] = target[taxon] - in_test[taxon]
    remaining = [rec_id for rec_id in counts if rec_id not in side]
    remaining.sort(key=lambda rec_id: (-len(counts[rec_id]), -size[rec_id], tie_break(rec_id)))
    for rec_id in remaining:
        filled = 0.0
        exceeded = 0.0
        for taxon, number in counts[rec_id].items():
            still_missing = max(0.0, missing[taxon])
            filled += min(number, still_missing)
            exceeded += max(0.0, number - still_missing)
        if filled > 0 and filled >= exceeded:
            side[rec_id] = "test"
            for taxon, number in counts[rec_id].items():
                in_test[taxon] += number
                missing[taxon] -= number
        else:
            side[rec_id] = "train"
    return side


# ---------------------------------------------------------------- identification examples
def taxon_generator(side, taxon):
    """Return the random generator of one taxon and side."""
    digest = hashlib.blake2b(f"{SEED}:{side}:{taxon}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def choose_examples(chunks, side_of_recording):
    """Return {side: {taxon: [(chunk_id, event index), ...]}}: at most 500 vocalizations per
    taxon and side, in random order. The position in the list is the rank of the example."""
    pools = {"train": defaultdict(list), "test": defaultdict(list)}
    for chunk in chunks:
        side = side_of_recording[chunk["rec_id"]]
        for index, event in enumerate(chunk["events"]):
            if event["taxon"] is not None:
                pools[side][event["taxon"]].append((chunk["chunk_id"], index))
    chosen = {}
    for side in ("train", "test"):
        chosen[side] = {}
        for taxon, pool in pools[side].items():
            pool = list(pool)
            taxon_generator(side, taxon).shuffle(pool)
            chosen[side][taxon] = pool[:EXAMPLES_PER_TAXON]
    return chosen


def taxon_table(chunks, chosen):
    """Return {taxon: {"group", "scorable"}}.

    The group of a taxon is that of the last corpus in which it occurs, except for the mammals
    of the bird soundscapes."""
    recordings_of = defaultdict(set)
    group = {}
    for chunk in chunks:
        for event in chunk["events"]:
            taxon = event["taxon"]
            if taxon is None:
                continue
            recordings_of[taxon].add(chunk["rec_id"])
            group[taxon] = corpora.GROUP_OF_TAXON.get(taxon, corpora.GROUP[chunk["corpus"]])
    table = {}
    for taxon in group:
        train_examples = len(chosen["train"].get(taxon, []))
        test_examples = len(chosen["test"].get(taxon, []))
        scorable = (len(recordings_of[taxon]) >= 2
                    and taxon != UNIDENTIFIED
                    and test_examples >= MIN_TEST_EXAMPLES
                    and train_examples >= MIN_TRAIN_EXAMPLES)
        table[taxon] = {"group": group[taxon], "scorable": scorable}
    return table


def build():
    """Apply all rules to the original corpora.

    Returns a dictionary with "chunks", "longest" ({corpus: longest duration kept}), "side"
    ({rec_id: "train" or "test"}), "examples" (see choose_examples) and "taxa" (see
    taxon_table)."""
    chunks, longest = build_chunks(corpora.read_all())
    side = split_recordings(chunks)
    examples = choose_examples(chunks, side)
    return {
        "chunks": chunks,
        "longest": longest,
        "side": side,
        "examples": examples,
        "taxa": taxon_table(chunks, examples),
    }
