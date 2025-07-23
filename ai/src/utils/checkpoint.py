from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup"""

    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        global_step: int,
        best_loss: float,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        batch_idx: int = 0,
        **kwargs,
    ) -> Path:
        """Save checkpoint and cleanup old ones"""
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

        if scaler:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        filepath = self.save_dir / f"checkpoint_epoch_{epoch + 1}_step_{global_step}.pt"
        torch.save(checkpoint, filepath)

        self._cleanup_old_checkpoints()
        return filepath

    def _cleanup_old_checkpoints(self) -> None:
        """Keep only the latest N checkpoints"""
        checkpoints = list(self.save_dir.glob("checkpoint_*.pt"))

        if len(checkpoints) > self.max_checkpoints:
            checkpoints.sort(key=lambda x: x.stat().st_mtime)

            for checkpoint in checkpoints[: -self.max_checkpoints]:
                checkpoint.unlink()

    def load_checkpoint(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if lr_scheduler and "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        if scaler and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return checkpoint
