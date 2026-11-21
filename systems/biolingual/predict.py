"""Identification predictions of the fine-tuned BioLingual model.

    python -m systems.biolingual.predict --data DIR --size 500

Reads models/biolingual/train_<size>.pt and writes
predictions/biolingual/identification_<size>.json. The model runs on the processor.
"""
import argparse

import torch
from transformers import ClapModel, ClapProcessor

from benchmark import config, data
from systems.biolingual import common
from systems.frozen_encoders import clips

BATCH = 32
THREADS = 8
WORKERS = 4


@torch.no_grad()
def predict(size, examples):
    """Return the predicted taxon of every example, in the given order."""
    taxon_names = common.taxa()
    processor = ClapProcessor.from_pretrained(common.MODEL_NAME)
    clap = ClapModel.from_pretrained(common.MODEL_NAME)
    model = common.Classifier(clap, len(taxon_names))
    model.load_state_dict(torch.load(common.model_file(size), map_location="cpu"))
    model.eval()
    batches = torch.utils.data.DataLoader(
        common.Clips(examples, taxon_names, augment=False),
        batch_size=BATCH,
        shuffle=False,
        num_workers=WORKERS,
        collate_fn=common.Collate(processor),
        worker_init_fn=common.limit_worker_threads,
    )
    predicted = []
    for inputs, _ in batches:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            classes = model(inputs).argmax(-1).tolist()
        for taxon_class in classes:
            predicted.append(taxon_names[taxon_class])
        print(f"{len(predicted)}/{len(examples)}", flush=True)
    return predicted


def main():
    """Predict the test examples with the model of one training size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--size", type=int, required=True, choices=config.SIZES)
    args = parser.parse_args()
    config.use_data_root(args.data)
    torch.set_num_threads(THREADS)
    examples = clips.examples("test")
    predicted = predict(args.size, examples)
    example_ids = [data.example_id(example) for example in examples]
    output_file = config.prediction_file("biolingual", "identification", args.size)
    data.write_json(output_file, dict(zip(example_ids, predicted)))
    print(output_file)


if __name__ == "__main__":
    main()
