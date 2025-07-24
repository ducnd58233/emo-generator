import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common.constants import (
    DEFAULT_CONTEXT_DIM,
    DEFAULT_LATENT_CHANNELS,
    DEFAULT_TIME_DIM,
    GROUPNORM_GROUPS,
    TIME_EMBEDDING_MULTIPLIER,
)
from .attention import CrossAttention, SelfAttention


class TimeEmbedding(nn.Module):
    """Time embedding layer for diffusion timesteps."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.proj1 = nn.Linear(n_embd, TIME_EMBEDDING_MULTIPLIER * n_embd)
        self.proj2 = nn.Linear(
            TIME_EMBEDDING_MULTIPLIER * n_embd, TIME_EMBEDDING_MULTIPLIER * n_embd
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.proj1(x))
        return self.proj2(x)


class UNETResidualBlock(nn.Module):
    """Residual block with time embedding integration."""

    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int = DEFAULT_TIME_DIM
    ):
        super().__init__()
        self.gn_feature = nn.GroupNorm(GROUPNORM_GROUPS, in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.time_embedding_proj = nn.Linear(time_dim, out_channels)

        self.gn_merged = nn.GroupNorm(GROUPNORM_GROUPS, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        self.residual_connection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(
        self, input_feature: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        residual = input_feature

        h = F.silu(self.gn_feature(input_feature))
        h = self.conv_feature(h)

        time_emb_processed = F.silu(time_emb)
        time_emb_projected = (
            self.time_embedding_proj(time_emb_processed).unsqueeze(-1).unsqueeze(-1)
        )

        merged_feature = h + time_emb_projected
        merged_feature = F.silu(self.gn_merged(merged_feature))
        merged_feature = self.conv_merged(merged_feature)

        return merged_feature + self.residual_connection(residual)


class UNETAttentionBlock(nn.Module):
    """Attention block with self-attention, cross-attention, and feed-forward."""

    def __init__(
        self, num_heads: int, head_dim: int, context_dim: int = DEFAULT_CONTEXT_DIM
    ):
        super().__init__()
        embed_dim = num_heads * head_dim

        # Input projection
        self.gn_in = nn.GroupNorm(GROUPNORM_GROUPS, embed_dim, eps=1e-6)
        self.proj_in = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # Attention layers
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn_1 = SelfAttention(num_heads, embed_dim, in_proj_bias=False)

        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn_2 = CrossAttention(
            num_heads, embed_dim, context_dim, in_proj_bias=False
        )

        # Feed-forward network
        self.ln_3 = nn.LayerNorm(embed_dim)
        self.ffn_geglu = nn.Linear(embed_dim, TIME_EMBEDDING_MULTIPLIER * embed_dim * 2)
        self.ffn_out = nn.Linear(TIME_EMBEDDING_MULTIPLIER * embed_dim, embed_dim)

        # Output projection
        self.proj_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(
        self, input_tensor: torch.Tensor, context_tensor: torch.Tensor
    ) -> torch.Tensor:
        skip_connection = input_tensor
        B, C, H, W = input_tensor.shape

        # Prepare input for attention
        h = self.proj_in(self.gn_in(input_tensor))
        h = h.view(B, C, H * W).transpose(-1, -2)

        # Self-attention with residual
        h = h + self.attn_1(self.ln_1(h))

        # Cross-attention with residual
        h = h + self.attn_2(self.ln_2(h), context_tensor)

        h_residual = h
        h = self.ln_3(h)
        intermediate, gate = self.ffn_geglu(h).chunk(2, dim=-1)
        h = self.ffn_out(intermediate * F.gelu(gate)) + h_residual

        h = h.transpose(-1, -2).view(B, C, H, W)
        return self.proj_out(h) + skip_connection


class Upsample(nn.Module):
    """Upsampling layer with nearest neighbor interpolation."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(feature_map, scale_factor=2, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    """Downsampling layer with strided convolution."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    """Sequential module that routes inputs based on layer type."""

    def forward(
        self,
        x: torch.Tensor,
        guidance_context: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        for module in self:
            match module:
                case UNETAttentionBlock():
                    x = module(x, guidance_context)
                case UNETResidualBlock():
                    x = module(x, time_embedding)
                case _:
                    x = module(x)
        return x


class UNET(nn.Module):
    """U-Net architecture for diffusion models."""

    def __init__(
        self, h_dim: int = 384, n_head: int = 8, time_dim: int = DEFAULT_TIME_DIM
    ):
        super().__init__()
        head_dim = h_dim // n_head

        self.conv_in = nn.Conv2d(
            DEFAULT_LATENT_CHANNELS, h_dim, kernel_size=3, padding=1
        )

        # Encoder path
        self.down_blocks = nn.ModuleList(
            [
                SwitchSequential(
                    UNETResidualBlock(h_dim, h_dim, time_dim),
                    UNETAttentionBlock(n_head, head_dim),
                    UNETResidualBlock(h_dim, h_dim, time_dim),
                )
            ]
        )

        # Bottleneck
        self.bottleneck = SwitchSequential(
            UNETResidualBlock(h_dim, h_dim, time_dim),
            UNETAttentionBlock(n_head, head_dim),
            UNETResidualBlock(h_dim, h_dim, time_dim),
        )

        # Decoder path
        self.up_blocks = nn.ModuleList(
            [
                SwitchSequential(
                    UNETResidualBlock(h_dim * 2, h_dim, time_dim),
                    UNETAttentionBlock(n_head, head_dim),
                    UNETResidualBlock(h_dim, h_dim, time_dim),
                )
            ]
        )

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        # Initial convolution
        x = self.conv_in(latent)
        skip_connections = [x]

        # Encoder path
        for down_block in self.down_blocks:
            x = down_block(x, context, time)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x, context, time)

        # Decoder path
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = up_block(x, context, time)

        return x


class UNETOutputLayer(nn.Module):
    """Output layer for U-Net with normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(GROUPNORM_GROUPS, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.silu(self.gn(x)))
