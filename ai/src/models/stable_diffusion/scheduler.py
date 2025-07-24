import math

import numpy as np
import torch

from ...common.constants import (
    DEFAULT_BETA_END,
    DEFAULT_BETA_START,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    TIME_EMBEDDING_FREQ_BASE,
)


def embed_timesteps(
    timesteps: torch.Tensor, embedding_dim: int = DEFAULT_EMBEDDING_DIM
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -math.log(TIME_EMBEDDING_FREQ_BASE)
        * torch.arange(half_dim, dtype=torch.float32)
        / half_dim
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DDPMScheduler:
    """DDPM (Denoising Diffusion Probabilistic Models) scheduler for training and inference."""

    def __init__(
        self,
        random_generator: torch.Generator,
        train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
        beta_start: float = DEFAULT_BETA_START,
        beta_end: float = DEFAULT_BETA_END,
    ):
        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, train_timesteps, dtype=torch.float32
            )
            ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one_val = torch.tensor(1.0)
        self.generator = random_generator
        self.total_train_timesteps = train_timesteps
        self.timesteps = torch.from_numpy(np.arange(0, train_timesteps)[::-1].copy())

    def set_steps(self, num_inference_steps: int = 50) -> None:
        """Set the number of inference steps and corresponding timesteps."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.total_train_timesteps // num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def _get_prev_timestep(self, timestep: int) -> int:
        """Get the previous timestep for a given timestep."""
        return timestep - self.total_train_timesteps // self.num_inference_steps

    def _get_variance(self, timestep: int) -> torch.Tensor:
        """Calculate variance for a given timestep."""
        prev_t = self._get_prev_timestep(timestep)
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        alpha_cumprod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one_val
        )
        beta_t = 1 - alpha_cumprod_t / alpha_cumprod_t_prev
        variance = (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * beta_t
        return torch.clamp(variance, min=1e-20)

    def step(
        self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor
    ) -> torch.Tensor:
        """Perform one denoising step."""
        t = timestep
        prev_t = self._get_prev_timestep(t)

        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one_val
        )
        beta_cumprod_t = 1 - alpha_cumprod_t
        beta_cumprod_t_prev = 1 - alpha_cumprod_t_prev
        alpha_t = alpha_cumprod_t / alpha_cumprod_t_prev
        beta_t = 1 - alpha_t

        # Predict original sample
        pred_original_sample = (
            latents - beta_cumprod_t**0.5 * model_output
        ) / alpha_cumprod_t**0.5

        # Compute coefficients
        pred_original_sample_coeff = (
            alpha_cumprod_t_prev**0.5 * beta_t
        ) / beta_cumprod_t
        current_sample_coeff = alpha_t**0.5 * beta_cumprod_t_prev / beta_cumprod_t

        # Compute predicted previous sample μ_t
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * latents
        )

        # Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype,
            )
            variance = (self._get_variance(t) ** 0.5) * noise

        return pred_prev_sample + variance

    def add_noise(
        self, original_samples: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to original samples at given timesteps."""
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_cumprod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(
            sqrt_alpha_cumprod.shape[0], *([1] * (original_samples.ndim - 1))
        )

        sqrt_one_minus_alpha_cumprod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(
            sqrt_one_minus_alpha_cumprod.shape[0], *([1] * (original_samples.ndim - 1))
        )

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        noisy_samples = (
            sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        )
        return noisy_samples, noise
