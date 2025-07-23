import torch
import torch.nn as nn

from .scheduler import embed_timesteps
from .unet import UNET, TimeEmbedding, UNETOutputLayer


class StableDiffusion(nn.Module):
    def __init__(self, h_dim: int = 384, n_head: int = 8, time_dim: int = 1280):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET(h_dim, n_head, time_dim)
        self.unet_output = UNETOutputLayer(h_dim, 4)

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        time_emb = embed_timesteps(timestep, embedding_dim=320).to(latent.device)
        time_emb = self.time_embedding(time_emb)

        output = self.unet(latent, context, time_emb)

        return self.unet_output(output)
