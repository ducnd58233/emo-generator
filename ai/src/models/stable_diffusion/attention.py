import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, use_causal_mask: bool = False) -> torch.Tensor:
        b, s, d = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(b, s, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(b, s, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(b, s, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)

        if use_causal_mask:
            causal_mask = torch.triu(torch.ones_like(qk, dtype=torch.bool), diagonal=1)
            qk = qk.masked_fill(causal_mask, -torch.inf)

        attn_weights = F.softmax(qk, dim=-1)
        attn_values = torch.matmul(attn_weights, v)

        attn_values = attn_values.permute(0, 2, 1, 3).contiguous().view(b, s, d)
        return self.out_proj(attn_values)


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        query_dim: int,
        context_dim: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = query_dim // num_heads

        self.query_map = nn.Linear(query_dim, query_dim, bias=in_proj_bias)
        self.key_map = nn.Linear(context_dim, query_dim, bias=in_proj_bias)
        self.value_map = nn.Linear(context_dim, query_dim, bias=in_proj_bias)
        self.output_map = nn.Linear(query_dim, query_dim, bias=out_proj_bias)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b_q, s_q, d_q = query.shape
        _, s_kv, _ = context.shape

        q = self.query_map(query)
        k = self.key_map(context)
        v = self.value_map(context)

        q = q.view(b_q, s_q, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(b_q, s_kv, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(b_q, s_kv, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn_weights = F.softmax(qk, dim=-1)
        attn_values = torch.matmul(attn_weights, v)

        attn_values = attn_values.permute(0, 2, 1, 3).contiguous().view(b_q, s_q, d_q)
        return self.output_map(attn_values)
