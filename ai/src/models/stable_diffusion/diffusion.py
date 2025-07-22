import torch
import torch.nn as nn

from .scheduler import embed_timesteps
from .unet import UNET, TimeEmbedding, UNETOutputLayer


class StableDiffusion(nn.Module):
    def __init__(self, h_dim: int = 384, n_head: int = 8):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET(h_dim, n_head)
        self.unet_output = UNETOutputLayer(h_dim, 4)

    @torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True, cache_enabled=True
    )
    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        time_emb = embed_timesteps(timestep).to(latent.device)
        time_emb = self.time_embedding(time_emb)

        output = self.unet(latent, context, time_emb)

        return self.unet_output(output)
