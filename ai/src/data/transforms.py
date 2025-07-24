from __future__ import annotations

from torchvision import transforms

from ..common.constants import DEFAULT_IMAGE_SIZE


def get_transforms(
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE, is_training: bool = True
) -> transforms.Compose:
    """Get image transforms for training or validation.

    Args:
        image_size: Target image size as (height, width)
        is_training: Whether transforms are for training (enables augmentation)

    Returns:
        Composed transforms
    """
    base_transforms = [
        transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    if is_training:
        # Add data augmentation for training
        augment_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
        ]
        final_transforms = (
            [augment_transforms[0]]
            + base_transforms[:1]
            + augment_transforms[1:]
            + base_transforms[1:]
        )
        return transforms.Compose(final_transforms)

    return transforms.Compose(base_transforms)
