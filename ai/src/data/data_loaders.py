from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from .datasets import EmojiDataset
from .transforms import get_train_transforms, get_val_transforms


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    dataset = EmojiDataset(
        data_dirs=config["data"]["data_dirs"],
        transform=get_train_transforms(config["data"]["image_size"]),
    )

    train_size = int(config["data"]["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_dataset.dataset.transform = get_val_transforms(
        tuple(config["data"]["image_size"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"],
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
