import torch
from torch import nn
import torch.nn.functional as F


class AttentionWithoutKVCache(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        # print(f"q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")

        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale, enable_gqa=True)
        # print(f"o.shape: {o.shape}")

        o = o.transpose(1, 2).squeeze(0)
        # after tranposing, the underlying tensor is not contiguous in memory, call contiguous() to get a contiguous copy
        o = o.contiguous().view(-1, self.num_heads * self.head_dim)
        # print(f"o.shape: {o.shape}")
        return o


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # print(f"q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # add a fake batch dimension
        # (seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        # print(f"q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")

        if self.k_cache is not None and self.v_cache is not None:
            k = torch.cat((self.k_cache, k), dim=2)
            v = torch.cat((self.v_cache, v), dim=2)
        self.k_cache, self.v_cache = k, v

        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale, enable_gqa=True)
        # print(f"o.shape: {o.shape}")

        o = o.transpose(1, 2).squeeze(0)
        # after tranposing, the underlying tensor is not contiguous in memory, call contiguous() to get a contiguous copy
        o = o.contiguous().view(-1, self.num_heads * self.head_dim)
        # print(f"o.shape: {o.shape}")
        return o
