import os

from typing import Tuple

import torch

from torchsummary import summary
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, random_split, Dataset


def get_train_set(root: str, input_size: int,
                  train_size: float, batch_size: int,
                  seed: int, cuda: bool, num_workers=1
                  ) -> Tuple[DataLoader, DataLoader]:
    # img -- reduce to 20% --> upsample to 224x224 -> Blur with ksize (10, 10)
    # Make female samples equal male

    # Use page-locked or not
    pin_memory = True if cuda is True else False

    dataset: Dataset = ImageFolder(root=os.path.join(root, "train"),
                                   transform=v2.Compose([
                                       # img from celeb A: 178 x 218 x 3
                                       v2.Resize(size=(int(178 * .2), int(218 * .2)), interpolation=InterpolationMode.NEAREST),
                                       v2.Resize(size=(input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                                       v2.GaussianBlur(kernel_size=5),
                                       v2.PILToTensor(),
                                       v2.ToDtype(torch.float32, scale=True)
                                   ]))

    train_set, validation_set = random_split(dataset=dataset,
                                             generator=torch.Generator().manual_seed(seed),
                                             lengths=[round(len(dataset) * train_size),
                                                      len(dataset) - round(len(dataset) * train_size)
                                                      ])

    train_set = DataLoader(dataset=train_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory
                           )

    validation_set = DataLoader(dataset=validation_set,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=pin_memory
                                )
    return train_set, validation_set


def get_test_set(root: str, input_size: int, batch_size: int, cuda: bool, num_workers=1) -> DataLoader:
    # Use page-locked or not
    pin_memory = True if cuda is True else False

    test_set: Dataset = ImageFolder(root=os.path.join(root, "test"),
                                    transform=v2.Compose([
                                        v2.Resize(size=(input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                                        v2.PILToTensor(),
                                        v2.ToDtype(torch.float32, scale=True)
                                    ]))

    test_set: DataLoader = DataLoader(dataset=test_set,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory
                                      )
    return test_set


def get_model_summary(model: torch.nn.Module, input_size: Tuple):
    return summary(model, input_size)
