import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class EmojiDataset(Dataset):
    def __init__(
        self,
        data_dirs: List[str],
        transform: Optional[Callable] = None,
    ):
        dataframes = []
        for data_dir in data_dirs:
            csv_file = os.path.join(data_dir, "metadata.csv")
            image_folder = os.path.join(data_dir, "images")
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.full_image_paths[idx]
        title = self.prompts[idx].replace('"', "").replace("'", "")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, title
