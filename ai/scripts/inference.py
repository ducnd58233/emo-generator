from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.inference.generator import EmojiGenerator
from src.utils.config import load_config
from src.utils.logging import get_logger

from ai.src.common.constants import (
    DEFAULT_LATENT_HEIGHT,
    DEFAULT_LATENT_WIDTH,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
)

sys.path.append(str(Path(__file__).parent.parent))

logger = get_logger(__name__)


def main() -> None:
    """Main inference function for generating emoji images."""
    parser = argparse.ArgumentParser(
        description="Generate Emojis with Stable Diffusion"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model checkpoint"
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
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--latent-height",
        type=int,
        default=DEFAULT_LATENT_HEIGHT,
        help="Height of latent space",
    )
    parser.add_argument(
        "--latent-width",
        type=int,
        default=DEFAULT_LATENT_WIDTH,
        help="Width of latent space",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for generation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for inference"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Validate model path
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

        # Load generator
        logger.info(f"Loading model from: {args.model_path}")
        generator = EmojiGenerator.from_pretrained(args.model_path, config, args.device)

        logger.info(f"Generating {args.num_images} images for prompt: '{args.prompt}'")

        # Generate images
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

                # Generate filename
                filename = (
                    f"emoji_seed_{seed}.png"
                    if args.num_images == 1
                    else f"emoji_{i+1}_seed_{seed}.png"
                )

                filepath = os.path.join(args.output_dir, filename)
                image.save(filepath)
                logger.info(f"Saved: {filepath}")

            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {e}")
                continue

        logger.info("Generation completed successfully!")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
