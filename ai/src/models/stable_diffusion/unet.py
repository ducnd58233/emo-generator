import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.proj1 = nn.Linear(n_embd, 4 * n_embd)
        self.proj2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x)
        x = F.silu(x)
        x = self.proj2(x)
        return x


class UNETResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_dim: int = 1280
    ):
        super().__init__()
        self.gn_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.time_embedding_proj = nn.Linear(time_dim, out_channels)

        self.gn_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_connection = nn.Identity()
        else:
            self.residual_connection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, input_feature: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = input_feature

        h = self.gn_feature(input_feature)
        h = F.silu(h)
        h = self.conv_feature(h)

        time_emb_processed = F.silu(time_emb)
        time_emb_projected = self.time_embedding_proj(time_emb_processed)
        time_emb_projected = time_emb_projected.unsqueeze(-1).unsqueeze(-1)

        merged_feature = h + time_emb_projected
        merged_feature = self.gn_merged(merged_feature)
        merged_feature = F.silu(merged_feature)
        merged_feature = self.conv_merged(merged_feature)

        return merged_feature + self.residual_connection(residual)


class UNETAttentionBlock(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, context_dim: int = 512):
        super().__init__()
        embed_dim = num_heads * head_dim

        self.gn_in = nn.GroupNorm(32, embed_dim, eps=1e-6)
        self.proj_in = nn.Conv2d(embed_dim, embed_dim,
                                 kernel_size=1, padding=0)

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn_1 = SelfAttention(num_heads, embed_dim, in_proj_bias=False)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn_2 = CrossAttention(
            num_heads, embed_dim, context_dim, in_proj_bias=False)
        self.ln_3 = nn.LayerNorm(embed_dim)

        self.ffn_geglu = nn.Linear(embed_dim, 4 * embed_dim * 2)
        self.ffn_out = nn.Linear(4 * embed_dim, embed_dim)
        self.proj_out = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=1, padding=0)

    def forward(self, input_tensor: torch.Tensor, context_tensor: torch.Tensor) -> torch.Tensor:
        skip_connection = input_tensor

        B, C, H, W = input_tensor.shape
        HW = H * W

        h = self.gn_in(input_tensor)
        h = self.proj_in(h)
        h = h.view(B, C, HW).transpose(-1, -2)

        # Self-attention
        attn1_skip = h
        h = self.ln_1(h)
        h = self.attn_1(h)
        h = h + attn1_skip

        # Cross-attention
        attn2_skip = h
        h = self.ln_2(h)
        h = self.attn_2(h, context_tensor)
        h = h + attn2_skip

        # Feed-forward
        ffn_skip = h
        h = self.ln_3(h)
        intermediate, gate = self.ffn_geglu(h).chunk(2, dim=-1)
        h = intermediate * F.gelu(gate)
        h = self.ffn_out(h)
        h = h + ffn_skip

        h = h.transpose(-1, -2).view(B, C, H, W)
        return self.proj_out(h) + skip_connection


class Upsample(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels,
                              kernel_size=3, padding=1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(feature_map, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, guidance_context: torch.Tensor,
                time_embedding: torch.Tensor) -> torch.Tensor:
        for module in self:
            if isinstance(module, UNETAttentionBlock):
                x = module(x, guidance_context)
            elif isinstance(module, UNETResidualBlock):
                x = module(x, time_embedding)
            else:
                x = module(x)
        return x


class UNET(nn.Module):
    def __init__(self, h_dim: int = 384, n_head: int = 8):
        super().__init__()
        # Simplified U-Net architecture for 32x32 images

        # Initial conv
        self.conv_in = nn.Conv2d(4, h_dim, kernel_size=3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList([
            SwitchSequential(
                UNETResidualBlock(h_dim, h_dim),
                UNETAttentionBlock(n_head, h_dim // n_head),
            ),
            SwitchSequential(
                UNETResidualBlock(h_dim, h_dim * 2),
                nn.Conv2d(h_dim * 2, h_dim * 2,
                          kernel_size=3, stride=2, padding=1),
            ),
        ])

        # Bottleneck
        self.bottleneck = SwitchSequential(
            UNETResidualBlock(h_dim * 2, h_dim * 2),
            UNETAttentionBlock(n_head, (h_dim * 2) // n_head),
            UNETResidualBlock(h_dim * 2, h_dim * 2),
        )

        # Decoder
        self.up_blocks = nn.ModuleList([
            SwitchSequential(
                UNETResidualBlock(h_dim * 4, h_dim * 2),
                UNETResidualBlock(h_dim * 2, h_dim),
                Upsample(h_dim),
            ),
            SwitchSequential(
                UNETResidualBlock(h_dim * 2, h_dim),
                UNETAttentionBlock(n_head, h_dim // n_head),
                UNETResidualBlock(h_dim, h_dim),
            ),
        ])

    def forward(self, latent: torch.Tensor, context: torch.Tensor,
                time: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.conv_in(latent)

        # Store skip connections
        skip_connections = [x]

        # Encoder
        for down_block in self.down_blocks:
            x = down_block(x, context, time)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x, context, time)

        # Decoder
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = up_block(x, context, time)

        return x


class UNETOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gn(x)
        x = F.silu(x)
        return self.conv(x)
