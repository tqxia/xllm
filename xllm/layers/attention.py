import torch
from torch import nn
import torch.nn.functional as F

from xllm.forward_context import ForwardContext, get_forward_context


class AttentionWithoutKVCache(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        max_seq_len,
        max_num_seqs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

    def forwar_with_unbatched_loop(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        ctx = get_forward_context()
        cu_seqlens = ctx.cu_seqlens
        orig_q, orig_k, orig_v = q, k, v

        res = []
        for idx in range(len(cu_seqlens)-1):
            q = orig_q[cu_seqlens[idx]:cu_seqlens[idx+1], :]
            k = orig_k[cu_seqlens[idx]:cu_seqlens[idx+1], :]
            v = orig_v[cu_seqlens[idx]:cu_seqlens[idx+1], :]

            cu_seqlen = cu_seqlens[idx+1] - cu_seqlens[idx]
            # print(f"Processing batch {idx}, cu_seqlen {cu_seqlen} total {len(cu_seqlens)-1}")
            q = q.view(cu_seqlen, self.num_heads, self.head_dim)
            k = k.view(cu_seqlen, self.num_kv_heads, self.head_dim)
            v = v.view(cu_seqlen, self.num_kv_heads, self.head_dim)

            q, k, v = (x.transpose(0, 1) for x in (q, k, v))
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale, enable_gqa=True)
            o = o.transpose(0, 1).reshape(cu_seqlen, -1)
            res.append(o)
        o = torch.cat(res, 0)
        return o

    def forward_with_block_diagonal_mask(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        ctx = get_forward_context()
        cu_seqlens = ctx.cu_seqlens

        # Manually build a block-diagonal causal mask
        assert cu_seqlens[-1] == q.shape[0]
        N = cu_seqlens[-1]
        mask = torch.zeros((N, N), dtype=torch.bool, device=q.device)

        for idx in range(len(cu_seqlens)-1):
            # Create a causal mask for current segment
            start = cu_seqlens[idx]
            l = cu_seqlens[idx+1] - cu_seqlens[idx]
            # reshape [l] to [l, 1]
            rows = torch.arange(start, start + l, device=q.device)[:, None]
            # reshape [l] to [1, l]
            cols = torch.arange(start, start + l, device=q.device)[None, :]
            # broadcast rows and compare with cols to create causal mask of [l, l]
            causal_block = (rows >= cols)
            # For example, the full mask for two sequences (lengths 3 and 2) among N=5 tokens:
            # 1 0 0 0 0
            # 1 1 0 0 0
            # 1 1 1 0 0
            # 0 0 0 1 0
            # 0 0 0 1 1
            mask[start:start + l, start:start + l] = causal_block

        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        q, k, v = (x.transpose(0, 1) for x in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale, enable_gqa=True)

        o = o.transpose(0, 1).reshape(N, -1)
        return o

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return self.forward_with_block_diagonal_mask(q, k, v)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        max_seq_len,
        max_num_seqs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.num_queries_per_kv = num_heads // num_kv_heads
        # FIXME: overwrite max_seq_len for test purpose
        max_seq_len = 1024
        self.k_cache = torch.zeros(max_num_seqs, max_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16).cuda(non_blocking=True)
        self.v_cache = torch.zeros(max_num_seqs, max_seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16).cuda(non_blocking=True)

    def forward_prefill(self, ctx: ForwardContext, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        cu_seqlens = ctx.cu_seqlens

        # Manually build a block-diagonal causal mask
        assert cu_seqlens[-1] == q.shape[0]
        N = cu_seqlens[-1]

        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        mask = torch.zeros((N, N), dtype=torch.bool, device=q.device)
        for idx in range(len(cu_seqlens)-1):
            # Create a causal mask for current segment
            start = cu_seqlens[idx]
            l = cu_seqlens[idx+1] - cu_seqlens[idx]
            # reshape [l] to [l, 1]
            rows = torch.arange(start, start + l, device=q.device)[:, None]
            # reshape [l] to [1, l]
            cols = torch.arange(start, start + l, device=q.device)[None, :]
            # broadcast rows and compare with cols to create causal mask of [l, l]
            causal_block = (rows >= cols)
            # For example, the full mask for two sequences (lengths 3 and 2) among N=5 tokens:
            # 1 0 0 0 0
            # 1 1 0 0 0
            # 1 1 1 0 0
            # 0 0 0 1 0
            # 0 0 0 1 1
            mask[start:start + l, start:start + l] = causal_block

            self.k_cache[idx, :l, :, :] = k[start:start + l, :, :]
            self.v_cache[idx, :l, :, :] = v[start:start + l, :, :]

        q, k, v = (x.transpose(0, 1) for x in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale, enable_gqa=True)

        o = o.transpose(0, 1).reshape(N, -1)
        return o

    def forward_decode(self, ctx: ForwardContext, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context_lens = ctx.context_lens

        orig_q = q.view(-1, self.num_heads, self.head_dim)
        orig_k = k.view(-1, self.num_kv_heads, self.head_dim)
        orig_v = v.view(-1, self.num_kv_heads, self.head_dim)

        res = []
        for idx in range(len(context_lens)):
            q = orig_q[idx].unsqueeze(0)

            seq_len = context_lens[idx]
            self.k_cache[idx, seq_len:seq_len+1, :, :] = orig_k[idx]
            self.v_cache[idx, seq_len:seq_len+1, :, :] = orig_v[idx]

            k = self.k_cache[idx, :seq_len+1, :, :]
            v = self.v_cache[idx, :seq_len+1, :, :]

            q, k, v = (x.transpose(0, 1) for x in (q, k, v))
            o = F.scaled_dot_product_attention(q, k, v, scale=self.scale, enable_gqa=True)
            o = o.transpose(0, 1).reshape(1, -1)
            res.append(o)

        o = torch.cat(res, 0)
        return o
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        ctx = get_forward_context()

        if ctx.is_prefill:
            return self.forward_prefill(ctx, q, k, v)
        else:
            return self.forward_decode(ctx, q, k, v)

    # def forward_single_batch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    #     q_len = q.size(0)
    #     kv_len = k.size(0)

    #     q = q.view(1, q_len, self.num_heads, self.head_dim)
    #     k = k.view(1, kv_len, self.num_kv_heads, self.head_dim)
    #     v = v.view(1, kv_len, self.num_kv_heads, self.head_dim)

    #     self.k_cache[:, self.cur_pos:self.cur_pos+kv_len, :, :] = k
    #     self.v_cache[:, self.cur_pos:self.cur_pos+kv_len, :, :] = v
    #     self.cur_pos += kv_len

    #     k = self.k_cache[:, :self.cur_pos, :, :]
    #     v = self.v_cache[:, :self.cur_pos, :, :]

    #     k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
    #     v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

    #     # F.scaled_dot_product_attention seems to be not working properly for decoding,
    #     # so here we opt for a manual implementation.
    #     attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)
    #     mask = torch.ones(1, q_len, kv_len, dtype=torch.bool, device=q.device).tril(diagonal=0)
    #     attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
    #     attn_probs = F.softmax(attn_scores, dim=-1)
    #     o = torch.einsum('bhqk,bkhd->bhqd', attn_probs, v)

    #     o = o.transpose(1, 2).squeeze(0)
    #     o = o.reshape(q_len, -1)
    #     return o
