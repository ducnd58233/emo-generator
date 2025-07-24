from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

from ...common.constants import DEFAULT_SCALING_FACTOR, VAE_MODEL_ID


class VAEEncoder(nn.Module):
    """VAE encoder for encoding images to latent space and decoding back."""

    def __init__(self, config: dict[str, str | bool | float]):
        super().__init__()
        model_id = config.get("model_id", VAE_MODEL_ID)
        low_cpu_mem_usage = config.get("low_cpu_mem_usage", True)

        self.vae = AutoencoderKL.from_pretrained(
            model_id, low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.scaling_factor = config.get("scaling_factor", DEFAULT_SCALING_FACTOR)

        # Freeze VAE parameters
        self.vae.requires_grad_(False)
        self.vae.eval()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space.

        Args:
            images: Input images tensor

        Returns:
            Encoded latent representations
        """
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            return latents * self.scaling_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to image space.

        Args:
            latents: Latent representations

        Returns:
            Decoded images
        """
        with torch.no_grad():
            latents = latents / self.scaling_factor
            return self.vae.decode(latents).sample
