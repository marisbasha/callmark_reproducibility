"""Add xeno-canto vocalizations in their own recordings to a manifest of mixtures.

    python -m systems.whisperid.pretraining.build_replay_manifest \
        --mixtures FILE --vocalizations FILE --out FILE

The output holds every mixture and half as many vocalizations without background, drawn
with replacement from the second manifest, in random order.
"""
import argparse
import random

SEED = 42


def main():
    """Write the combined manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixtures", required=True)
    parser.add_argument("--vocalizations", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.mixtures) as handle:
        lines = handle.readlines()
    with open(args.vocalizations) as handle:
        vocalizations = handle.readlines()
    generator = random.Random(SEED)
    for _ in range(len(lines) // 2):
        lines.append(generator.choice(vocalizations))
    generator.shuffle(lines)
    with open(args.out, "w") as handle:
        handle.writelines(lines)
    print(f"{len(lines)} lines", flush=True)


if __name__ == "__main__":
    main()
