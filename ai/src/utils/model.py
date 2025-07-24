import os
import random

import numpy as np
import torch
import torch.nn as nn

from ..common.constants import DEFAULT_SEED
from .logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic behavior for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


def get_device() -> str:
    """Get the best available device for computation.

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {gpu_name} ({gpu_count} GPU(s) available)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = "cpu"
        logger.info("Using CPU device")

    return device


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total parameters, trainable parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate the approximate model size in megabytes.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    return (param_size + buffer_size) / (1024**2)
