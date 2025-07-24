from __future__ import annotations

import os
from collections.abc import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class EmojiDataset(Dataset):
    """Dataset for emoji generation with text prompts."""

    def __init__(
        self,
        data_dirs: list[str],
        transform: Callable | None = None,
    ):
        """Initialize emoji dataset.

        Args:
            data_dirs: List of directories containing metadata.csv and images/
            transform: Optional image transforms to apply
        """
        dataframes = []
        for data_dir in data_dirs:
            csv_file = os.path.join(data_dir, "metadata.csv")
            image_folder = os.path.join(data_dir, "images")

            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"Metadata file not found: {csv_file}")
            if not os.path.exists(image_folder):
                raise FileNotFoundError(f"Images folder not found: {image_folder}")

            df = pd.read_csv(csv_file)
            df["image_path"] = df["file_name"].astype(str).str.replace("\\", "/")
            df["full_image_path"] = df["image_path"].apply(
                lambda x: os.path.join(image_folder, x)
            )
            dataframes.append(df)

        self.dataframe = pd.concat(dataframes, ignore_index=True)
        self.transform = transform
        self.prompts = self.dataframe["prompt"].tolist()
        self.full_image_paths = self.dataframe["full_image_path"].tolist()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """Get item by index.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (image_tensor, prompt_text)
        """
        image_path = self.full_image_paths[idx]
        title = self.prompts[idx].replace('"', "").replace("'", "")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}") from e

        if self.transform:
            image = self.transform(image)

        return image, title
