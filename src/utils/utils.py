import os
import re

from typing import Tuple
from json import JSONDecoder, JSONDecodeError

import torch

from torchsummary import summary
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, random_split

NOT_WHITESPACE = re.compile(r'\S')


def json_decoder(document: str, pos=0, decoder=JSONDecoder()):
    """
    Acceptable format for document:
        a/ Format 1: Single json obj
            '''{obj}'''
        b/ Format 2: Multiple json objs
            '''{obj_1},
               {obj_2},
               {obj_3}
            '''
        c/ Format 2: List of Single or Multiple json objs
            '''
            [{obj_1},
               {obj_2},
               {obj_3}]
            '''
    """
    while True:
        match = NOT_WHITESPACE.search(document, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError as e:
            print(e)
            # do something sensible if there's some error
            raise
        yield obj


def get_dataset(root: str, img_size: int,
                train_size: float, batch_size: int,
                seed: int, cuda: bool, num_workers=1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # img -- reduce to 20% --> upsample to 224x224 -> Blur with ksize (10, 10)
    # Make female samples equal male

    # Use page-locked or not
    pin_memory = True if cuda is True else False

    # Train & Validation
    train_set = ImageFolder(root=os.path.join(root, "train"),
                            transform=v2.Compose([
                                # img from celeb A: 178 x 218 x 3
                                v2.Resize(size=(int(178 * .2), int(218 * .2)), interpolation=InterpolationMode.NEAREST),
                                v2.Resize(size=(img_size, img_size), interpolation=InterpolationMode.BICUBIC),
                                v2.GaussianBlur(kernel_size=5),
                                v2.PILToTensor(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))

    train_set, validation_set = random_split(dataset=train_set,
                                             generator=torch.Generator().manual_seed(seed),
                                             lengths=[round(len(train_set) * train_size),
                                                      len(train_set) - round(len(train_set) * train_size)
                                                      ]
                                             )

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

    # Test
    test_set = ImageFolder(root=os.path.join(root, "test"),
                           transform=v2.Compose([
                               v2.Resize(size=(int(178 * .2), int(218 * .2)), interpolation=InterpolationMode.NEAREST),
                               v2.Resize(size=(img_size, img_size), interpolation=InterpolationMode.BICUBIC),
                               v2.GaussianBlur(kernel_size=5),
                               v2.PILToTensor(),
                               v2.ToDtype(torch.float32, scale=True),
                               v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                           ]))

    test_set = DataLoader(dataset=test_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          pin_memory=pin_memory
                          )
    return train_set, validation_set, test_set


def get_model_summary(model: torch.nn.Module, input_size: Tuple):
    return summary(model, input_size)
