from pathlib import Path
from typing import Any

import torch

from ..common.constants import CHECKPOINT_EXT, DEFAULT_MAX_CHECKPOINTS
from .logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup and versioning."""

    def __init__(
        self, save_dir: str | Path, max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS
    ):
        """Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        global_step: int,
        best_loss: float,
        scaler: torch.cuda.amp.GradScaler | None = None,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> Path:
        """Save checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            lr_scheduler: Learning rate scheduler state
            epoch: Current epoch
            global_step: Global training step
            best_loss: Best validation loss so far
            scaler: Mixed precision scaler (optional)
            batch_idx: Current batch index for resuming
            **kwargs: Additional data to save

        Returns:
            Path to saved checkpoint file
        """
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "batch_idx": batch_idx,
            **kwargs,
        }

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        filename = f"checkpoint_epoch_{epoch + 1}_step_{global_step}{CHECKPOINT_EXT}"
        filepath = self.save_dir / filename

        torch.save(checkpoint, filepath)

        self._cleanup_old_checkpoints()
        return filepath

    def _cleanup_old_checkpoints(self) -> None:
        """Keep only the most recent N checkpoints."""
        checkpoints = list(self.save_dir.glob(f"checkpoint_*{CHECKPOINT_EXT}"))

        if len(checkpoints) <= self.max_checkpoints:
            return

        checkpoints.sort(key=lambda x: x.stat().st_mtime)

        for checkpoint in checkpoints[: -self.max_checkpoints]:
            try:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint.name}")
            except OSError as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint.name}: {e}")

    def load_checkpoint(
        self,
        filepath: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
        device: str = "cuda",
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load checkpoint and restore training state.

        Args:
            filepath: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            lr_scheduler: Learning rate scheduler to load state into (optional)
            scaler: Mixed precision scaler to load state into (optional)
            device: Device to map tensors to
            strict: Whether to strictly enforce that the keys match

        Returns:
            Checkpoint dictionary with metadata

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        try:
            checkpoint = torch.load(filepath, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {filepath}: {e}") from e

        # Load model state
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # Load scaler state
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return checkpoint

    def get_latest_checkpoint(self) -> Path | None:
        """Get the path to the most recent checkpoint.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = list(self.save_dir.glob(f"checkpoint_*{CHECKPOINT_EXT}"))

        if not checkpoints:
            return None

        return max(checkpoints, key=lambda x: x.stat().st_mtime)

    def list_checkpoints(self) -> list[Path]:
        """List all available checkpoints.

        Returns:
            List of checkpoint paths sorted by modification time (newest first)
        """
        checkpoints = list(self.save_dir.glob(f"checkpoint_*{CHECKPOINT_EXT}"))
        return sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)
