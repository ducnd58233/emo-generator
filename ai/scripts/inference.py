import argparse
import os
import sys
from pathlib import Path

from src.inference.generator import EmojiGenerator
from src.utils.config import load_config
from src.utils.logging import get_logger

sys.path.append(str(Path(__file__).parent.parent))

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Emojis with Stable Diffusion"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for generation"
    )
    parser.add_argument(
        "--num_images", type=int, default=1, help="Number of images to generate"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--latent_height", type=int, default=4, help="Height of latent space"
    )
    parser.add_argument(
        "--latent_width", type=int, default=4, help="Width of latent space"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for generation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for inference"
    )

    args = parser.parse_args()

    try:
        # Load config
        logger.info("Loading configuration...")
        config = load_config(args.config)

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Check if model exists
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

        logger.info(f"Loading model from: {args.model_path}")
        generator = EmojiGenerator.from_pretrained(args.model_path, config, args.device)

        logger.info(f"Generating {args.num_images} images for prompt: '{args.prompt}'")

        # Generate multiple images if requested
        for i in range(args.num_images):
            try:
                logger.info(f"Generating image {i+1}/{args.num_images}...")

                # Use different seed for each image
                seed = args.seed + i

                image = generator.generate(
                    prompt=args.prompt,
                    num_inference_steps=args.num_steps,
                    latent_height=args.latent_height,
                    latent_width=args.latent_width,
                    seed=seed,
                )

                # Save image
                if args.num_images == 1:
                    filename = f"emoji_seed_{seed}.png"
                else:
                    filename = f"emoji_{i+1}_seed_{seed}.png"

                filepath = os.path.join(args.output_dir, filename)
                image.save(filepath)
                logger.info(f"Saved: {filepath}")

            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {e}")
                continue

        logger.info("Generation complete!")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
