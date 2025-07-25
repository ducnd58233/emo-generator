from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split

from .datasets import EmojiDataset
from .transforms import get_transforms


def create_data_loaders(
    config: dict[str, dict[str, str | int | float | bool]], seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    dataset = EmojiDataset(
        data_dirs=config["data"]["data_dirs"],
        transform=get_transforms(config["data"]["image_size"], is_training=True),
    )

    train_size = int(config["data"]["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Set validation transforms
    val_dataset.dataset.transform = get_transforms(
        tuple(config["data"]["image_size"]), is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"],
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"],
    )

    return train_loader, val_loader
