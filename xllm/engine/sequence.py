from copy import copy
from enum import Enum

from xllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


class Sequence:
    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_prompt_tokens = len(token_ids)
        self.num_tokens = len(token_ids)

        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens

    def __len__(self):
        return self.num_tokens
    
    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED
    
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
