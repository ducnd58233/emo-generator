import argparse
import os
import sys

from src.inference.generator import EmojiGenerator
from src.utils.config import load_config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
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

    # Load config
    config = load_config(args.config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    generator = EmojiGenerator.from_pretrained(args.model_path, config, args.device)

    print(f"Generating {args.num_images} images for prompt: '{args.prompt}'")

    images = generator.generate(
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        batch_size=args.num_images,
    )

    # Save images
    for i, image in enumerate(images):
        filename = f"emoji_{i+1}_seed_{args.seed}.png"
        filepath = os.path.join(args.output_dir, filename)
        image.save(filepath)
        print(f"Saved: {filepath}")

    print("Generation complete!")


if __name__ == "__main__":
    main()
