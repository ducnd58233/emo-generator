import torch
import torch.nn as nn

from ...common.constants import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_H_DIM,
    DEFAULT_LATENT_CHANNELS,
    DEFAULT_N_HEAD,
    DEFAULT_TIME_DIM,
)
from .scheduler import embed_timesteps
from .unet import UNET, TimeEmbedding, UNETOutputLayer


class StableDiffusion(nn.Module):
    """Stable Diffusion model for text-to-image generation."""

    def __init__(
        self,
        h_dim: int = DEFAULT_H_DIM,
        n_head: int = DEFAULT_N_HEAD,
        time_dim: int = DEFAULT_TIME_DIM,
    ):
        super().__init__()
        self.time_embedding = TimeEmbedding(DEFAULT_EMBEDDING_DIM)
        self.unet = UNET(h_dim, n_head, time_dim)
        self.unet_output = UNETOutputLayer(h_dim, DEFAULT_LATENT_CHANNELS)

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the Stable Diffusion model.

        Args:
            latent: Latent tensor of shape (B, C, H, W)
            context: Context tensor from text encoder of shape (B, seq_len, embed_dim)
            timestep: Timestep tensor of shape (B,)

        Returns:
            Predicted noise tensor of same shape as latent
        """
        time_emb = embed_timesteps(timestep).to(latent.device)
        time_emb = self.time_embedding(time_emb)

        # Run through U-Net
        noise_pred = self.unet(latent, context, time_emb)

        # Output layer
        return self.unet_output(noise_pred)
