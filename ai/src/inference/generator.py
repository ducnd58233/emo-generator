from __future__ import annotations

import torch
from PIL import Image
from tqdm import tqdm

from ..common.constants import (
    DEFAULT_LATENT_CHANNELS,
    DEFAULT_LATENT_HEIGHT,
    DEFAULT_LATENT_WIDTH,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
)
from ..models.encoders.clip import CLIPTextEncoder
from ..models.encoders.vae import VAEEncoder
from ..models.stable_diffusion.diffusion import StableDiffusion
from ..models.stable_diffusion.scheduler import DDPMScheduler
from ..utils.checkpoint import load_checkpoint_file
from ..utils.image import tensor_to_pil
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EmojiGenerator:
    """Emoji generator using trained Stable Diffusion model"""

    def __init__(
        self,
        diffusion_model: StableDiffusion,
        vae_encoder: VAEEncoder,
        text_encoder: CLIPTextEncoder,
        scheduler: DDPMScheduler,
        device: str = "cuda",
    ):
        self.diffusion_model = diffusion_model
        self.vae_encoder = vae_encoder
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.device = device

        # Set models to evaluation mode
        self.diffusion_model.eval()
        self.vae_encoder.eval()
        self.text_encoder.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: dict[str, dict[str, str | int | float]],
        device: str = "cuda",
    ) -> EmojiGenerator:
        """Load generator from pretrained checkpoint"""
        logger.info("Loading EmojiGenerator from checkpoint...")

        # Initialize models
        model_config = config["model"]

        diffusion_model = StableDiffusion(
            h_dim=model_config["stable_diffusion"]["h_dim"],
            n_head=model_config["stable_diffusion"]["n_head"],
            time_dim=model_config["stable_diffusion"]["time_dim"],
        ).to(device)

        text_encoder = CLIPTextEncoder(config=model_config["clip"], device=device)

        vae_encoder = VAEEncoder(config=model_config["vae"]).to(device)

        # Initialize scheduler with generator
        generator = torch.Generator(device=device)
        scheduler = DDPMScheduler(
            random_generator=generator,
            train_timesteps=model_config["stable_diffusion"]["num_train_timesteps"],
            beta_start=model_config["stable_diffusion"]["beta_start"],
            beta_end=model_config["stable_diffusion"]["beta_end"],
        )

        checkpoint = load_checkpoint_file(model_path, map_location=device)
        diffusion_model.load_state_dict(checkpoint["model_state_dict"])

        logger.info("Loaded model weights from checkpoint")

        generator_instance = cls(
            diffusion_model=diffusion_model,
            vae_encoder=vae_encoder,
            text_encoder=text_encoder,
            scheduler=scheduler,
            device=device,
        )

        logger.info("EmojiGenerator loaded successfully")
        return generator_instance

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        latent_height: int = DEFAULT_LATENT_HEIGHT,
        latent_width: int = DEFAULT_LATENT_WIDTH,
        seed: int = DEFAULT_SEED,
    ) -> Image.Image:
        """Generate emoji image from text prompt"""

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        with torch.no_grad():
            text_embeddings = self.text_encoder([prompt])
            logger.info(f"Text embeddings shape: {text_embeddings.shape}")

            # Create initial noise
            latents = torch.randn(
                (1, DEFAULT_LATENT_CHANNELS, latent_height, latent_width),
                device=self.device,
                dtype=torch.float32,
            )
            logger.info(f"Initial latents shape: {latents.shape}")

            # Set scheduler timesteps
            self.scheduler.set_steps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            logger.info(f"Number of timesteps: {len(timesteps)}")

            # Denoising loop
            for i, timestep in enumerate(tqdm(timesteps, desc="Generating")):
                timestep_tensor = torch.tensor([timestep], device=self.device)

                # Predict noise
                noise_pred = self.diffusion_model(
                    latents, text_embeddings, timestep_tensor
                )

                # Scheduler step
                latents = self.scheduler.step(timestep, latents, noise_pred)

            # Decode latents to image
            images = self.vae_encoder.decode(latents)

            # Convert to PIL Image
            image = tensor_to_pil(images)

        return image

    def generate_batch(
        self,
        prompts: list[str],
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        latent_height: int = DEFAULT_LATENT_HEIGHT,
        latent_width: int = DEFAULT_LATENT_WIDTH,
        seed: int = DEFAULT_SEED,
    ) -> list[Image.Image]:
        """Generate multiple emoji images from a list of prompts"""

        images = []
        for i, prompt in enumerate(prompts):
            # Use different seed for each image
            image_seed = seed + i
            image = self.generate(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                latent_height=latent_height,
                latent_width=latent_width,
                seed=image_seed,
            )
            images.append(image)

        return images
