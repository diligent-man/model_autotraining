import os
import pandas as pd

from PIL import Image
from overrides import override

from torchvision.transforms import v2
from torch.utils.data import Dataset


__all__ = ["CelebADataset"]


class CelebADataset(Dataset):
    """Read images"""
    __dataset_path: str
    __annotation: pd.DataFrame
    __transform: v2.Transform

    def __init__(self, dataset_path: str, annotation: pd.DataFrame, transform=None):
        self.__dataset_path: str = dataset_path
        self.__annotation: pd.DataFrame = annotation
        self.__transform: v2.Transform = transform

    def __len__(self):
        return len(os.listdir(self.__dataset_path.__str__()))

    @override
    def __getitem__(self, idx: int):
        file_name: str = self.__annotation["image_id"][idx]
        img: Image = Image.open(fp=os.path.join(self.__dataset_path, file_name), mode="r").convert("RGB")

        if self.__transform:
            img = self.__transform(img)
        return img
