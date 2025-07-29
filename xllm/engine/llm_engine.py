from transformers import AutoConfig, AutoTokenizer

from xllm.config import Config
from xllm.sampling_params import SamplingParams
from xllm.engine.model_runner import ModelRunner
from xllm.engine.sequence import Sequence


class LLMEngine:
    def __init__(self, model):
        config = Config(model)
        config.hf_config = AutoConfig.from_pretrained(config.model)
        print("[DEBUG] hf_config: ", config.hf_config)
        config.max_model_len = min(config.max_model_len, config.hf_config.max_position_embeddings)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.config = config
        self.model_runner = ModelRunner(config)

    def step(self, seq: Sequence):
        output_token_id = self.model_runner.run(seq, seq.num_completion_tokens == 0)
        seq.append_token(output_token_id)
        if output_token_id == self.config.eos or seq.num_completion_tokens >= seq.max_tokens:
            seq.is_finished = True

    def generate(self, prompt: str, sampling_params: SamplingParams):
        input_token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(input_token_ids, sampling_params)

        while not seq.is_finished:
            self.step(seq)
        
        completion_token_ids = seq.completion_token_ids
        return {"text": self.tokenizer.decode(completion_token_ids), "token_ids": completion_token_ids}
