import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_model_len: int = 4096
    hf_config: AutoConfig | None = None
    eos: int = -1
    # Batching
    max_num_seqs: int = 16 # Max batch size
