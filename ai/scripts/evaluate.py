import argparse
import os
import sys
from pathlib import Path

import torch
from src.data.data_loaders import create_data_loaders
from src.inference.generator import EmojiGenerator
from src.training.trainer import StableDiffusionTrainer
from src.utils.config import load_config, merge_configs
from src.utils.logging import get_logger
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

logger = get_logger(__name__)


def evaluate_model(trainer, val_loader, device):
    """Evaluate model on validation set"""
    logger.info("Starting model evaluation...")
    trainer.diffusion_model.eval()
    total_loss = 0.0
    total_samples = 0

    try:
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Evaluating")

            for images, prompts in progress_bar:
                images = images.to(device)

                # Encode to latents
                latents = trainer.vae_encoder.encode(images)

                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    trainer.scheduler.total_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                )

                # Add noise
                noisy_latents, noise = trainer.scheduler.add_noise(latents, timesteps)

                # Encode text
                text_embeddings = trainer.text_encoder(prompts)

                # Predict noise
                noise_pred = trainer.diffusion_model(
                    noisy_latents, text_embeddings, timesteps
                )

                # Compute loss
                loss = trainer.criterion(noise_pred, noise)

                batch_loss = loss.item() * len(images)
                total_loss += batch_loss
                total_samples += len(images)

                # Update progress bar
                current_avg = total_loss / total_samples if total_samples > 0 else 0
                progress_bar.set_postfix(
                    {
                        "batch_loss": f"{loss.item():.5f}",
                        "avg_loss": f"{current_avg:.5f}",
                    }
                )

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    logger.info(f"Evaluation completed. Average loss: {avg_loss:.6f}")
    return avg_loss


def generate_sample_images(generator, config, output_dir="outputs"):
    """Generate sample images for qualitative evaluation"""
    logger.info("Generating sample images for qualitative evaluation...")

    sample_prompts = [
        "a happy blob emoji with sparkling eyes, green skin, purple background",
        "an angry blob emoji with narrowed eyes, orange skin, white background",
        "a confused blob emoji with wide eyes, yellow skin, dark blue background",
        "a sad blob emoji with droopy eyes, green skin, white background",
        "a grumpy blob emoji with half-closed eyes, purple skin, red background",
        "a happy blob emoji with teary eyes, pink skin, white background",
    ]

    os.makedirs(output_dir, exist_ok=True)

    # Use config defaults or fallbacks
    num_inference_steps = config.get("inference", {}).get("num_inference_steps", 50)
    latent_height = config.get("data", {}).get("latent_size", [4, 4])[0]
    latent_width = config.get("data", {}).get("latent_size", [4, 4])[1]

    for i, prompt in enumerate(sample_prompts):
        try:
            logger.info(
                f"Generating sample {i+1}/{len(sample_prompts)}: '{prompt[:50]}...'"
            )

            image = generator.generate(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                latent_height=latent_height,
                latent_width=latent_width,
                seed=42 + i,
            )

            output_path = os.path.join(output_dir, f"sample_{i+1}.png")
            image.save(output_path)
            logger.info(f"Saved sample to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to generate sample {i+1}: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            continue

    logger.info("Sample generation completed!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stable Diffusion Model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default="config/model.yaml", help="Path to model config"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="config/data.yaml",
        help="Path to data config",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="config/training.yaml",
        help="Path to training config",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save sample images",
    )
    parser.add_argument(
        "--skip_samples", action="store_true", help="Skip sample image generation"
    )
    parser.add_argument(
        "--skip_eval", action="store_true", help="Skip validation evaluation"
    )

    args = parser.parse_args()

    try:
        # Load and merge configs
        logger.info("Loading configuration files...")
        model_config = load_config(args.config)
        data_config = load_config(args.data_config)
        training_config = load_config(args.training_config)
        config = merge_configs(model_config, data_config, training_config)

        # Check if model exists
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

        # Validation evaluation
        if not args.skip_eval:
            logger.info("Creating data loaders...")
            _, val_loader = create_data_loaders(config)

            logger.info("Initializing trainer...")
            trainer = StableDiffusionTrainer(config, args.device, is_validation=True)

            logger.info(f"Loading model from: {args.model_path}")
            trainer.load_checkpoint(args.model_path)

            # Evaluate
            avg_loss = evaluate_model(trainer, val_loader, args.device)
            logger.info(f"Final average validation loss: {avg_loss:.6f}")

        # Generate sample images
        if not args.skip_samples:
            logger.info("Loading generator for sample image generation...")
            try:
                generator = EmojiGenerator.from_pretrained(
                    args.model_path, config, args.device
                )
                generate_sample_images(generator, config, args.output_dir)
            except Exception as e:
                logger.error(f"Failed to initialize generator: {e}")
                logger.error(
                    "You may need to run training first to create a proper checkpoint"
                )
        else:
            logger.info("Skipping sample image generation (--skip_samples flag used)")

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
