import argparse
import os
import sys
from typing import Optional

from src.data.data_loaders import create_data_loaders
from src.training.trainer import StableDiffusionTrainer
from src.utils.config import load_config, merge_configs
from src.utils.logging import get_logger
from src.utils.model import get_device, set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = get_logger(__name__)


def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint file in a directory.
    Args:
        save_dir: Directory to search.
    Returns:
        Path to latest checkpoint or None.
    """
    if not os.path.isdir(save_dir):
        return None
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith(".pt")]
    if not checkpoints:
        return None
    checkpoints = sorted(
        checkpoints, key=lambda x: os.path.getmtime(os.path.join(save_dir, x))
    )
    return os.path.join(save_dir, checkpoints[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Stable Diffusion for Emoji Generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="config/training.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="config/data.yaml",
        help="Path to data config file",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    model_config = load_config(args.config)
    train_config = load_config(args.train_config)
    data_config = load_config(args.data_config)
    config = merge_configs(model_config, train_config, data_config)
    device = args.device or get_device()
    logger.info(f"Using device: {device}")
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    trainer = StableDiffusionTrainer(config, device)

    resume_path = args.resume or find_latest_checkpoint(
        config["experiment"]["save_dir"]
    )
    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)
    else:
        logger.info("No checkpoint found, starting training from scratch.")
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
