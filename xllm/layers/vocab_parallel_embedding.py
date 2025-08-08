import torch
from torch import nn
import torch.nn.functional as F


class VocabEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        y = F.embedding(x, self.weight)
        return y


class LMHead(VocabEmbedding):

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
    ):
        super().__init__(vocab_size, hidden_size)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        # if x.shape[0] != 1:
        #     x = x[-1:]
        logits = F.linear(x, self.weight, self.bias)
        return logits
