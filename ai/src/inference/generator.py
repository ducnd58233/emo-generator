import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Union
from PIL import Image

from ..models.stable_diffusion.diffusion import StableDiffusion
from ..models.stable_diffusion.scheduler import DDPMScheduler
from ..models.encoders.clip import CLIPTextEncoder
from ..models.encoders.vae import VAEEncoder
from ..utils.image import rescale_tensor


class EmojiGenerator:
    def __init__(
        self,
        diffusion_model: StableDiffusion,
        text_encoder: CLIPTextEncoder,
        vae_encoder: VAEEncoder,
        scheduler: DDPMScheduler,
        device: str = "cuda"
    ):
        self.diffusion_model = diffusion_model.eval()
        self.text_encoder = text_encoder
        self.vae_encoder = vae_encoder
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 32,
        width: int = 32,
        seed: Optional[int] = None,
        batch_size: int = 1
    ) -> List[Image.Image]:

        # Set random seed
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        # Encode text
        prompts = [prompt] * batch_size
        text_embeddings = self.text_encoder(prompts)

        # Classifier-free guidance: create unconditional embeddings
        unconditional_prompts = [""] * batch_size
        uncond_embeddings = self.text_encoder(unconditional_prompts)

        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Set scheduler timesteps
        self.scheduler.set_steps(num_inference_steps)

        # Create random latents
        latents_shape = (batch_size, 4, height // 8, width // 8)
        latents = torch.randn(
            latents_shape, generator=generator, device=self.device)

        # Denoising loop
        for t in tqdm(self.scheduler.timesteps, desc="Generating"):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)

            # Predict noise
            noise_pred = self.diffusion_model(
                latent_model_input, text_embeddings, t)

            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = self.scheduler.step(t, latents, noise_pred)

        # Decode latents to images
        images = self.vae_encoder.decode(latents)

        # Convert to PIL Images
        images = rescale_tensor(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        pil_images = [Image.fromarray(img) for img in images]
        return pil_images

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: dict,
        device: str = "cuda"
    ) -> "EmojiGenerator":

        # Load models
        diffusion_model = StableDiffusion(
            h_dim=config['model']['stable_diffusion']['h_dim'],
            n_head=config['model']['stable_diffusion']['n_head']
        ).to(device)

        text_encoder = CLIPTextEncoder(
            model_id=config['model']['clip']['model_id'],
            device=device
        )

        vae_encoder = VAEEncoder(
            model_id=config['model']['vae']['model_id'],
            scaling_factor=config['model']['vae']['scaling_factor']
        ).to(device)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])

        # Create scheduler
        generator = torch.Generator(device=device)
        scheduler = DDPMScheduler(
            random_generator=generator,
            train_timesteps=config['model']['stable_diffusion']['num_train_timesteps'],
            beta_start=config['model']['stable_diffusion']['beta_start'],
            beta_end=config['model']['stable_diffusion']['beta_end']
        )

        return cls(diffusion_model, text_encoder, vae_encoder, scheduler, device)
