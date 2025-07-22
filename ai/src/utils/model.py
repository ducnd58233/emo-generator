import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model(model: nn.Module) -> None:
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = True


def get_device() -> str:
    """Get optimal device for training"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def print_model_info(model: nn.Module, name: str = "Model") -> None:
    """Print model information"""
    total_params = count_parameters(model)
    print(f"{name}: {total_params:,} trainable parameters")
