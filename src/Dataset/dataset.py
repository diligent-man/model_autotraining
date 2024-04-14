from .DefaultDataset import DefaultDataset
from .ImageFolderDataset import  ImageFolderDataset


__all__ = ["available_dataset"]


available_dataset = {
    "DefaultDataset": DefaultDataset,
    "ImageFolderDataset": ImageFolderDataset
}
