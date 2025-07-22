from typing import Dict

import torch
import torch.nn.functional as F


def compute_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def compute_mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def compute_training_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> Dict[str, float]:
    return {
        "mse_loss": compute_mse_loss(pred, target).item(),
        "mae_loss": compute_mae_loss(pred, target).item(),
    }
