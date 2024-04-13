from typing import Dict, Any

from src.Dataset import available_dataset
from src.DataLoader import available_dataloader

from src.open_src import available_transform, available_interpolation, available_dtype

from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader, random_split

__all__ = ['DataManager']


class DataManager:
    __seed: float = 12345
    __device: str = "cpu"

    __transform: Dict[str, Any] = None
    __target_transform: Dict[str, Any] = None


    def __init__(self,
                 seed: float = 12345,
                 device: str = "cpu",
                 transform: Dict[str, Dict] = None,
                 target_transform: Dict[str, Dict] = None
                 ):
        self.__seed = seed
        self.__device = device
        self.__transform = self.__get_transformation(transform)
        self.__target_transform = self.__get_transformation(target_transform)

    def get_dataloader(self,
                       dataset: str,
                       dataset_args: str,
                       dataloader: str,
                       dataloader_args: Dict[str, Any] = None,
                       ) -> DataLoader:
        assert dataloader in available_dataloader.keys(), "Your selected dataloader is unavailble"

        dataset = self.__get_dataset(dataset=dataset, dataset_args=dataset_args)
        dataloader = available_dataloader[dataloader](dataset=dataset, **dataloader_args)
        return dataloader
    ##########################################################################################################


    def __get_dataset(self, dataset: str, dataset_args: Dict[str, Any]) -> Dataset:
        # Default dataset
        # TODO: Extend to other kinds of datasets
        assert dataset in available_dataset.keys(), "Your selected dataset is unavailable"
        return available_dataset[dataset](transform=self.__transform, target_transform=self.__target_transform, **dataset_args)
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
