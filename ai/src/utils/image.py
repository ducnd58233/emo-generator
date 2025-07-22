from typing import Tuple

import numpy as np
import torch
from PIL import Image


def rescale_tensor(
    tensor: torch.Tensor,
    in_range: Tuple[float, float],
    out_range: Tuple[float, float],
    clamp: bool = True,
) -> torch.Tensor:
    """Rescale tensor values from input range to output range"""
    in_min, in_max = in_range
    out_min, out_max = out_range

    in_span = in_max - in_min
    out_span = out_max - out_min

    scaled = (tensor - in_min) / (in_span + 1e-8)
    rescaled = out_min + (scaled * out_span)

    if clamp:
        rescaled = torch.clamp(rescaled, out_min, out_max)

    return rescaled


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tensor = rescale_tensor(tensor, (-1, 1), (0, 255), clamp=True)
    array = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(array)


def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """Convert PIL Image to tensor"""
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    if normalize:
        tensor = tensor * 2.0 - 1.0  # Normalize to [-1, 1]

    return tensor
