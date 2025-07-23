import os
from typing import Any, Dict, Optional

import torch

from .logging import get_logger

logger = get_logger(__name__)


def delete_file(filepath: str) -> None:
    """
    Delete a file if it exists.
    Args:
        filepath (str): Path to the file to delete.
    """
    try:
        if os.path.isfile(filepath):
            os.remove(filepath)
            logger.info(f"Deleted file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to delete {filepath}: {e}")


def cleanup_old_checkpoints(directory: str, max_checkpoints: int = 1) -> None:
    """
    Keep only the latest N checkpoint files in a directory, delete older ones.
    Args:
        directory (str): Directory containing checkpoints.
        max_checkpoints (int): Number of most recent checkpoints to keep.
    """
    checkpoints = [f for f in os.listdir(directory) if f.endswith(".pt")]
    if len(checkpoints) > max_checkpoints:
        checkpoints = sorted(
            checkpoints, key=lambda x: os.path.getmtime(os.path.join(directory, x))
        )
        for ckpt in checkpoints[:-max_checkpoints]:
            delete_file(os.path.join(directory, ckpt))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    filepath: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    global_step: Optional[int] = None,
    best_loss: Optional[float] = None,
    max_checkpoints: int = 1,
    **kwargs: Any,
) -> None:
    """
    Save model checkpoint and keep only the latest N checkpoints in the directory.
    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        lr_scheduler: LR scheduler to save.
        epoch: Current epoch.
        loss: Current loss.
        filepath: Path to save checkpoint.
        scaler: GradScaler for mixed precision (optional).
        global_step: Current global step (optional).
        best_loss: Best validation loss so far (optional).
        max_checkpoints: Number of most recent checkpoints to keep.
        **kwargs: Additional state to save.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "loss": loss,
        **kwargs,
    }
    if global_step is not None:
        checkpoint["global_step"] = global_step
    if best_loss is not None:
        checkpoint["best_loss"] = best_loss
    if scaler:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint: {filepath}")
    cleanup_old_checkpoints(os.path.dirname(filepath), max_checkpoints)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Load model checkpoint and restore state.
    Args:
        filepath: Path to checkpoint file.
        model: Model to load state into.
        optimizer: Optimizer to load state into (optional).
        lr_scheduler: LR scheduler to load state into (optional).
        scaler: GradScaler to load state into (optional).
        device: Device to map checkpoint to.
    Returns:
        The loaded checkpoint dictionary.
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    logger.info(f"Loaded checkpoint: {filepath}")
    return checkpoint
