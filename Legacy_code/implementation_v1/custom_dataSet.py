import itertools
import os
import numpy as np
import pandas as pd

from typing import Tuple, Any
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split


class CustomImageDataset(Dataset):
    __img_dataset: pd.DataFrame
    __img_dit: str
    __train_size: float
    __target_transform: v2
    __transform: v2

    def __init__(self, annotations_file: str, img_dir: str,
                 train_size: float, transform=None, target_transform=None) -> None:
        self.__img_dataset: pd.DataFrame = pd.read_csv(annotations_file, header=0, names=["file_name", "label"])
        self.__img_dir: str = img_dir
        self.__train_size: float = train_size
        self.__target_transform = target_transform
        self.__transform = transform

    # Methods
    # Public
    def __len__(self):
        return len(self.__img_dataset)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.__img_dir, self.__img_dataset.iloc[index, 0])
        image = read_image(img_path)
        label = self.__img_dataset.iloc[index, 1]

        if self.__transform:
            image = self.__transform(image)

        if self.__target_transform:
            label = self.__target_transform(label)
        return image, label

    def get_train_test_indices(self, shuffle: bool = True, random_state: int = 123) -> Tuple[Any, Any]:
        train_indices, test_indices, _, _ = train_test_split(np.arange(self.__len__()),
                                                             self.__img_dataset["label"],
                                                             train_size=self.__train_size,
                                                             shuffle=shuffle,
                                                             stratify=self.__img_dataset["label"],
                                                             random_state=random_state)
        return train_indices, test_indices





