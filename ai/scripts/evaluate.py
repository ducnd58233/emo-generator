from src.utils.config import load_config, merge_configs
from src.data.data_loaders import create_data_loaders
from src.training.trainer import StableDiffusionTrainer
from tqdm import tqdm
import numpy as np
import torch
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_model(trainer, val_loader, device):
    """Evaluate model on validation set"""
    trainer.diffusion_model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, prompts in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)

            # Encode to latents
            latents = trainer.vae_encoder.encode(images)

            # Sample random timesteps
            timesteps = torch.randint(
                0, trainer.scheduler.total_train_timesteps,
                (latents.shape[0],), device=device
            )

            # Add noise
            noisy_latents, noise = trainer.scheduler.add_noise(
                latents, timesteps)

            # Encode text
            text_embeddings = trainer.text_encoder(prompts)

            # Predict noise
            noise_pred = trainer.diffusion_model(
                noisy_latents, text_embeddings, timesteps)

            # Compute loss
            loss = trainer.criterion(noise_pred, noise)

            total_loss += loss.item() * len(images)
            total_samples += len(images)

    avg_loss = total_loss / total_samples
    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Stable Diffusion Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                        help='Path to model config')
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml',
                        help='Path to data config')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Load configs
    model_config = load_config(args.config)
    data_config = load_config(args.data_config)
    config = merge_configs(model_config, data_config)

    # Create data loader
    _, val_loader = create_data_loaders(config)

    # Initialize trainer
    trainer = StableDiffusionTrainer(config, args.device)

    # Load checkpoint
    print(f"Loading model from: {args.model_path}")
    trainer.load_checkpoint(args.model_path)

    # Evaluate
    print("Evaluating model...")
    avg_loss = evaluate_model(trainer, val_loader, args.device)

    print(f"Average validation loss: {avg_loss:.6f}")

    # Generate sample images for qualitative evaluation
    from src.inference.generator import EmojiGenerator

    generator = EmojiGenerator.from_pretrained(
        args.model_path, config, args.device)

    sample_prompts = [
        "a happy smiling emoji",
        "a sad crying emoji",
        "a surprised emoji with wide eyes",
        "an angry red emoji"
    ]

    print("\nGenerating sample images...")
    os.makedirs("evaluation_samples", exist_ok=True)

    for i, prompt in enumerate(sample_prompts):
        images = generator.generate(
            prompt=prompt,
            num_inference_steps=50,
            seed=42 + i
        )
        images[0].save(f"evaluation_samples/sample_{i+1}.png")
        print(f"Generated sample for: '{prompt}'")


if __name__ == "__main__":
    main()
