"""Xeno-canto recordings that are part of the benchmark and stay out of pretraining."""
import re

from systems.whisperid import examples


def blocked_numbers():
    """Return every number of five to seven digits in a recording name of the benchmark.

    Xeno-canto recordings of the benchmark carry their catalogue number in their name."""
    numbers = set()
    for split in ("train_500", "test"):
        for example in examples.rows(split, with_audio=False):
            for match in re.finditer(r"(?:XC)?#?(\d{5,7})", example["rec_id"]):
                numbers.add(match.group(1))
    return numbers
