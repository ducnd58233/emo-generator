from src.utils.config import load_config, merge_configs
from src.utils.model import set_seed, get_device
from src.data.data_loaders import create_data_loaders
from src.training.trainer import StableDiffusionTrainer
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description='Train Stable Diffusion for Emoji Generation')
    parser.add_argument('--config', type=str, default='config/model.yaml',
                        help='Path to model config file')
    parser.add_argument('--train_config', type=str, default='config/training.yaml',
                        help='Path to training config file')
    parser.add_argument('--data_config', type=str, default='config/data.yaml',
                        help='Path to data config file')
    parser.add_argument('--device', type=str,
                        default=None, help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load configurations
    model_config = load_config(args.config)
    train_config = load_config(args.train_config)
    data_config = load_config(args.data_config)

    # Merge configurations
    config = merge_configs(model_config, train_config, data_config)

    # Set device
    device = args.device or get_device()
    print(f"Using device: {device}")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Initialize trainer
    trainer = StableDiffusionTrainer(config, device)

    # Load checkpoint if resuming
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
