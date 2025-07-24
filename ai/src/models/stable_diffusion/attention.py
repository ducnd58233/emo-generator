from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention mechanism for processing sequences."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_size)

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, use_causal_mask: bool = False) -> torch.Tensor:
        """Apply self-attention to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            use_causal_mask: Whether to apply causal masking

        Returns:
            Output tensor of same shape as input
        """
        b, s, d = x.shape

        # Generate Q, K, V projections
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(b, s, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.head_size).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if use_causal_mask:
            mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, -torch.inf)

        # Apply softmax and compute weighted values
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(attn_output)


class CrossAttention(nn.Module):
    """Cross-attention mechanism for conditioning on external context."""

    def __init__(
        self,
        num_heads: int,
        query_dim: int,
        context_dim: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_size = query_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_size)

        self.query_proj = nn.Linear(query_dim, query_dim, bias=in_proj_bias)
        self.key_proj = nn.Linear(context_dim, query_dim, bias=in_proj_bias)
        self.value_proj = nn.Linear(context_dim, query_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=out_proj_bias)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention between query and context.

        Args:
            query: Query tensor of shape (batch_size, query_len, query_dim)
            context: Context tensor of shape (batch_size, context_len, context_dim)

        Returns:
            Output tensor of same shape as query
        """
        b_q, s_q, d_q = query.shape
        _, s_kv, _ = context.shape

        # Generate projections
        q = self.query_proj(query)
        k = self.key_proj(context)
        v = self.value_proj(context)

        # Reshape for multi-head attention
        q = q.view(b_q, s_q, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(b_q, s_kv, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(b_q, s_kv, self.num_heads, self.head_size).transpose(1, 2)

        # Compute attention scores and apply softmax
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(b_q, s_q, d_q)
        return self.out_proj(attn_output)
