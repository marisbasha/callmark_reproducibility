"""Run the WhisperSeg training script with the validation decoding used for DualWhisper.

Started by train.py inside the WhisperSeg code folder; takes the arguments of WhisperSeg's
train.py. During validation the decoder prompt ends with the species token that starts every
training target, and the end-of-text token is allowed directly after the prompt, as in
prediction.
"""
import copy
import os
import runpy
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import model as whisperseg_model  # noqa: E402

START_TOKENS = ["<|startoftranscript|>", "<|en|>", "<|notimestamps|>"]


def generate_segment_text(
    self,
    sliced_audios_features,
    batch_size,
    max_length,
    num_beams,
    top_k=1,
    top_p=1.0,
    length_penalty=1.0,
    status_monitor=None,
):
    """Return the text decoded for every spectrogram window (validation during training)."""
    species_codebook = getattr(self.model.config, "species_codebook", {}) or {}
    species_token = species_codebook.get("unknown", "<|unknown|>")
    prompt_tokens = START_TOKENS + self.tokenizer.tokenize(species_token)
    prompt = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
    generation_config = copy.deepcopy(self.model.generation_config)
    generation_config.begin_suppress_tokens = None

    generated_text_list = []
    for position in range(0, len(sliced_audios_features), batch_size):
        batch = sliced_audios_features[position:position + batch_size]
        features = np.asarray([item[2] for item in batch])
        input_features = torch.from_numpy(features).to(self.device)
        decoder_input_ids = torch.LongTensor([prompt] * len(batch)).to(self.device)
        generated_ids = self.model.generate(
            inputs=input_features,
            decoder_input_ids=decoder_input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=num_beams == 1,
            top_k=top_k,
            top_p=top_p,
            generation_config=generation_config,
            length_penalty=length_penalty,
        )
        generated_text_list += self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )
    return generated_text_list


whisperseg_model.WhisperSegmenterForEval.generate_segment_text = generate_segment_text
sys.argv = ["train.py"] + sys.argv[1:]
runpy.run_path("train.py", run_name="__main__")
