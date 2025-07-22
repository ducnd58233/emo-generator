from torchvision import transforms
from typing import Tuple


def get_train_transforms(image_size: Tuple[int, int] = (32, 32)) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_val_transforms(image_size: Tuple[int, int] = (32, 32)) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
