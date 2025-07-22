import torch
import os
from typing import Dict, Any, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    filepath: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    **kwargs
) -> None:
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss,
        **kwargs
    }

    if scaler:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint
