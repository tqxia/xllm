import torch

from xllm.config import Config
from xllm.engine.sequence import Sequence
from xllm.layers.sampler import Sampler
from xllm.models.qwen3 import Qwen3ForCausalLM
from xllm.utils.loader import load_model


class ModelRunner:
    def __init__(self, config: Config):
        self.config = config
        hf_config = config.hf_config

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def prepare(self, seq: Sequence):
        seqlen = len(seq)
        input_ids = seq[:]
        positions = list(range(0, seqlen))
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        return input_ids_tensor, positions_tensor

    def prepare_prefill(self, seq: Sequence):
        seqlen = len(seq)
        input_ids = seq[:]
        positions = list(range(0, seqlen))
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        return input_ids_tensor, positions_tensor

    def prepare_decode(self, seq: Sequence):
        input_ids = [seq.last_token]
        positions = [len(seq)-1]
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        return input_ids_tensor, positions_tensor
    
    def prepare_temperature(self, seq: Sequence):
        temperatures = torch.tensor([seq.temperature], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def run(self, seq: Sequence, is_prefill: bool) -> int:
        input_ids, positions = self.prepare_prefill(seq) if is_prefill else self.prepare_decode(seq)
        # input_ids, positions = self.prepare(seq)
        temperatures = self.prepare_temperature(seq)
        logits = self.model.compute_logits(self.model(input_ids, positions))
        token_ids = self.sampler(logits, temperatures).tolist()
        return token_ids[-1]
