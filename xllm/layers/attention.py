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
        max_seq_len,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q_len = q.size(0)
        kv_len = k.size(0)

        q = q.view(1, q_len, self.num_heads, self.head_dim)
        k = k.view(1, kv_len, self.num_kv_heads, self.head_dim)
        v = v.view(1, kv_len, self.num_kv_heads, self.head_dim)

        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale, enable_gqa=True)

        o = o.transpose(1, 2).squeeze(0)
        return o.reshape(q_len, -1)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        max_seq_len,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.k_cache = torch.zeros(1, max_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16).cuda()
        self.v_cache = torch.zeros(1, max_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16).cuda()
        self.cur_pos = 0

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q_len = q.size(0)
        kv_len = k.size(0)

        q = q.view(1, q_len, self.num_heads, self.head_dim)
        k = k.view(1, kv_len, self.num_kv_heads, self.head_dim)
        v = v.view(1, kv_len, self.num_kv_heads, self.head_dim)

        self.k_cache[:, self.cur_pos:self.cur_pos+kv_len, :, :] = k
        self.v_cache[:, self.cur_pos:self.cur_pos+kv_len, :, :] = v
        self.cur_pos += kv_len

        k = self.k_cache[:, :self.cur_pos, :, :]
        v = self.v_cache[:, :self.cur_pos, :, :]

        k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
        v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

        # F.scaled_dot_product_attention seems to be not working properly for decoding,
        # so here we opt for a manual implementation.
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)
        mask = torch.ones(1, q_len, kv_len, dtype=torch.bool, device=q.device).tril(diagonal=0)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        o = torch.einsum('bhqk,bkhd->bhqd', attn_probs, v)

        o = o.transpose(1, 2).squeeze(0)
        o = o.reshape(q_len, -1)
        return o
