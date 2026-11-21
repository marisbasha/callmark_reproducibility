"""The WhisperID model: Whisper with a 250-frame encoder that generates the taxon name."""
import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer

from systems.whisperid import features


def load(checkpoint, device, dtype=torch.float32):
    """Return (model, tokenizer) of a checkpoint directory or Hugging Face model name.

    The positional table of the encoder is cut to the 250 frames of the 5-s input."""
    tokenizer = WhisperTokenizer.from_pretrained(checkpoint, language="english")
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=dtype)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    encoder = model.model.encoder
    positions = encoder.embed_positions.weight.detach()[:features.ENCODER_FRAMES].clone()
    table = torch.nn.Embedding(features.ENCODER_FRAMES, positions.shape[1])
    encoder.embed_positions = table.to(positions.dtype)
    with torch.no_grad():
        encoder.embed_positions.weight.copy_(positions)
    model.config.max_source_positions = features.ENCODER_FRAMES
    return model.to(device), tokenizer


def prompt_tokens(tokenizer, sampling_rate):
    """Return the decoder input that precedes the taxon name."""
    return tokenizer.encode(features.prompt(sampling_rate))[:-1]


def training_tokens(tokenizer, sampling_rate, taxon):
    """Return (decoder input, labels) of one training example; only the name is trained."""
    prompt = features.prompt(sampling_rate)
    tokens = tokenizer.encode(prompt + " " + taxon)
    prompt_length = len(tokenizer.encode(prompt))
    labels = tokens[1:]
    for position in range(max(0, prompt_length - 2)):
        labels[position] = -100
    return tokens[:-1], labels


def collate(pad_token):
    """Return the function that joins training examples into a batch."""

    def join(batch):
        length = max(len(example["decoder_input_ids"]) for example in batch)
        inputs = []
        labels = []
        for example in batch:
            missing = length - len(example["decoder_input_ids"])
            inputs.append(example["decoder_input_ids"] + [pad_token] * missing)
            labels.append(example["labels"] + [-100] * missing)
        stacked = torch.stack([torch.from_numpy(example["input_features"]) for example in batch])
        return {
            "input_features": stacked,
            "decoder_input_ids": torch.LongTensor(inputs),
            "labels": torch.LongTensor(labels),
        }

    return join


@torch.no_grad()
def generate_taxa(model, tokenizer, spectrograms, sampling_rate):
    """Return the taxon names generated greedily for a batch of one sampling rate."""
    prompt = prompt_tokens(tokenizer, sampling_rate)
    decoder_input = torch.LongTensor([prompt] * len(spectrograms)).to(model.device)
    generated = model.generate(
        spectrograms.to(model.device),
        decoder_input_ids=decoder_input,
        max_new_tokens=16,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    names = []
    for sequence in generated:
        text = tokenizer.decode(sequence[len(prompt):], skip_special_tokens=True)
        names.append(text.split("species:")[-1].strip())
    return names
