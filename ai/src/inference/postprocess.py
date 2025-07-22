from typing import List, Tuple

import torch
from PIL import Image


def postprocess_images(
    images: torch.Tensor, target_size: Tuple[int, int] = (512, 512)
) -> List[Image.Image]:
    """Post-process generated images"""

    # Rescale to [0, 255]
    images = (images + 1.0) * 127.5
    images = torch.clamp(images, 0, 255).to(torch.uint8)

    # Convert to PIL
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    pil_images = [Image.fromarray(img) for img in images]

    # Resize if needed
    if target_size != (images.shape[2], images.shape[1]):
        pil_images = [img.resize(target_size, Image.LANCZOS) for img in pil_images]

    return pil_images


def apply_filters(
    image: Image.Image, brightness: float = 1.0, contrast: float = 1.0
) -> Image.Image:
    """Apply brightness and contrast adjustments"""
    from PIL import ImageEnhance

    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)

    return image
