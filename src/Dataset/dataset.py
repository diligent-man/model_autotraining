from .DefaultDataset import DefaultDataset
from torchvision.datasets import (
    ImageFolder
)

__all__ = ["available_dataset"]


available_dataset = {
    "DefaultDataset": DefaultDataset,
    "ImageFolder": ImageFolder
}
