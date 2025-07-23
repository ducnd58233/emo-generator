from typing import Any, Dict

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class VAEEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            config["model_id"], low_cpu_mem_usage=bool(config["low_cpu_mem_usage"])
        )
        self.scaling_factor = float(config["scaling_factor"])

        # Freeze VAE parameters
        self.vae.requires_grad_(False)
        self.vae.eval()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            return latents * self.scaling_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = latents / self.scaling_factor
            return self.vae.decode(latents).sample
