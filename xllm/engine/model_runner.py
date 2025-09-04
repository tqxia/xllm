from typing import List, Tuple

import torch

from xllm.config import Config
from xllm.engine.sequence import Sequence
from xllm.forward_context import set_forward_context
from xllm.layers.sampler import Sampler
from xllm.models.qwen3 import Qwen3ForCausalLM
from xllm.utils.loader import load_model


class ModelRunner:
    def __init__(self, config: Config):
        self.config = config
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def prepare_inputs(self, seqs: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.prepare_inputs_for_prefill(seqs)

    def prepare_inputs_for_prefill(self, seqs: List[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens = [0]
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[:])
            positions.extend(range(0, seqlen))
            cu_seqlens.append(cu_seqlens[-1] + seqlen)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        set_forward_context(True, cu_seqlens=cu_seqlens)
        return input_ids_tensor, positions_tensor

    def prepare_inputs_for_decode(self, seqs: List[Sequence]):
        input_ids = []
        positions = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

        set_forward_context(False, context_lens=context_lens)
        return input_ids_tensor, positions_tensor
    
    def prepare_sample(self, seqs: List[Sequence]):
        temperatures = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def run(self, seqs: List[Sequence], is_prefill: bool) -> int:
        # input_ids, positions = self.prepare_inputs(seqs)
        input_ids, positions = self.prepare_inputs_for_prefill(seqs) if is_prefill else self.prepare_inputs_for_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        logits = self.model.compute_logits(self.model(input_ids, positions))
        token_ids = self.sampler(logits, temperatures).tolist()
        return token_ids
