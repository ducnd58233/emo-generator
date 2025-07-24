from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error loss."""
    return F.mse_loss(pred, target)


def compute_mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Mean Absolute Error loss."""
    return F.l1_loss(pred, target)


def compute_training_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """Compute training metrics for predicted and target tensors.

    Args:
        pred: Predicted tensor
        target: Target tensor

    Returns:
        Dictionary of computed metrics
    """
    return {
        "mse_loss": compute_mse_loss(pred, target).item(),
        "mae_loss": compute_mae_loss(pred, target).item(),
    }
