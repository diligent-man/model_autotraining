import os

from typing import Dict, Tuple, Any
from src.open_src import available_transform, available_interpolation, available_dtype

import torch

from torchvision.transforms import Compose
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split


class DataManager:
    __root: str
    __seed: float = 12345
    __device: str = "cpu"
    transform: Dict[str, Any] = None
    target_transform: Dict[str, Any] = None

    def __init__(self,
                 root: str,
                 seed: float = 12345,
                 device: str = "cpu",
                 transform: Dict[str, Dict] = None,
                 target_transform: Dict[str, Dict] = None
                 ):
        self.__root = root
        self.__seed = seed
        self.__device = device
        self.__transform = self.__get_transformation(transform)
        self.__target_transform = self.__get_transformation(target_transform)

    def get_train_val_loader(self,
                             train_size: float = 0.9,
                             dataloader_args: dict[str, Any] = None,
                             customDataloader: DataLoader = None
                             ) -> Tuple[DataLoader, DataLoader]:
        dataset = self.__get_dataset("train")

        train_size = round(len(dataset) * train_size)

        train_set, validation_set = random_split(dataset=dataset,
                                                 generator=torch.Generator().manual_seed(self.__seed),
                                                 lengths=[train_size, len(dataset) - train_size]
                                                 )

        if customDataloader is None:
            train_set = DataLoader(dataset=train_set, **dataloader_args)
            validation_set = DataLoader(dataset=validation_set, **dataloader_args)
        else:
            train_set = customDataloader(dataset=train_set, **dataloader_args)
            validation_set = customDataloader(dataset=validation_set, **dataloader_args)
        return train_set, validation_set

    def get_test_loader(self, dataloader_args: dict[str, Any], customDataloader: DataLoader = None) -> DataLoader:
        dataset = self.__get_dataset("test")
        if customDataloader:
            return customDataloader(dataset=dataset, **dataloader_args)
        else:
            return DataLoader(dataset=dataset, **dataloader_args)
    ##########################################################################################################


    def __get_dataset(self, dataset_type: str) -> Dataset:
        """
        Args:
            root: dataset dir
            transform: Dict of transformation name and its corresponding kwargs
            target_transform:                     //                            but for labels
        """
        return ImageFolder(os.path.join(self.__root, dataset_type), self.__transform, self.__target_transform)
    ##########################################################################################################


    @staticmethod
    def __get_transformation(transforms: Dict[str, Dict] = None) -> Compose:
        compose: Compose = Compose([])

        if transforms is not None:
            # Verify transformation
            for transform in transforms.keys():
                assert transform in available_transform.keys(), "Your selected transform method is unavailable"

                # Verify interpolation mode & replace str name to its corresponding func
                if transform in ("Resize", "RandomRotation"):
                    assert transforms[transform]["interpolation"] in available_interpolation.keys(), "Your selected interpolation mode in unavailable"
                    transforms[transform]["interpolation"] = available_interpolation[transforms[transform]["interpolation"]]

                # Verify dtype & replace str name to its corresponding func
                if transform in ("ToDtype"):
                    assert transforms[transform]["dtype"] in available_dtype.keys(), "Your selected dtype in unavailable"
                    transforms[transform]["dtype"] = available_dtype[transforms[transform]["dtype"]]

            compose = Compose([available_transform[transform](**args) for transform, args in transforms.items()])
        return compose


    # Code snippet for tracking how each class is drawn by dataloader
    # from operator import add
    # def print_class_counts(data_loader_dict):
    #     counter_lst = [0] * 2
    #     for epoch in range(3):
    #         for phase, data_loader in data_loader_dict.items():
    #             print(f"Phase: {phase}")
    #             for i, (inputs, labels) in enumerate(data_loader):
    #                 class_counts = labels.bincount()
    #                 # print(f"Batch {i + 1}: {class_counts.tolist()}")
    #
    #                 if len(class_counts.tolist()) < 2:
    #                     counter_lst = list( map(add, counter_lst, class_counts.tolist() + [0]))
    #                 else:
    #                     counter_lst = list(map(add, counter_lst, class_counts.tolist()))
    #                     print(counter_lst)
    #             print()
    #             print()
    #
    # print_class_counts({
    #     "train": self.__train_loader
    # })
