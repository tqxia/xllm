from typing import List, Union
from transformers import AutoConfig, AutoTokenizer

from xllm.config import Config
from xllm.sampling_params import SamplingParams
from xllm.engine.model_runner import ModelRunner
from xllm.engine.scheduler import Scheduler
from xllm.engine.sequence import Sequence, SequenceStatus


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
        self.scheduler = Scheduler(config)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        print(f"[DEBUG] seqs: {len(seqs)}, is_prefill: {is_prefill}")
        output_token_ids = self.model_runner.run(seqs, is_prefill)
        # process outputs
        outputs = []
        for (seq, token_id) in zip(seqs, output_token_ids):
            seq.append_token(token_id)
            if token_id == self.config.eos or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.scheduler.remove_seq(seq)

                completion_token_ids = seq.completion_token_ids
                outputs.append({
                    "text": self.tokenizer.decode(completion_token_ids),
                    "token_ids": completion_token_ids,
                })
        return outputs

    def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
    ):
        if isinstance(prompt, str):
            input_token_ids = self.tokenizer.encode(prompt)
        else:
            input_token_ids = prompt

        seq = Sequence(input_token_ids, sampling_params)
        self.scheduler.add_seq(seq)

    def has_unfinished_requests(self):
        return self.scheduler.has_unfinished_seqs()
