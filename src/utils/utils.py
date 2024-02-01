import os
import re
import torch

from typing import Tuple
from json import JSONDecoder, JSONDecodeError

from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import InterpolationMode


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
                seed: int, num_workers=1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Use page-locked or not
    pin_memory = True if torch.cuda.is_available() is True else False

    # Train & Validation
    train_set = ImageFolder(root=os.path.join(root, "train"),
                            transform=v2.Compose([
                                v2.Resize(size=(img_size, img_size),
                                          interpolation=InterpolationMode.NEAREST),
                                v2.PILToTensor(),
                                v2.ToDtype(torch.float32, scale=True)
                            ]))

    train_set, vaidation_set = random_split(dataset=train_set,
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

    validation_set = DataLoader(dataset=vaidation_set,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=pin_memory
                                )

    # Test
    test_set = ImageFolder(root=os.path.join(root, "test"),
                           transform=v2.Compose([
                               v2.Resize(size=(img_size, img_size),
                                         interpolation=InterpolationMode.NEAREST),
                               v2.PILToTensor(),
                               v2.ToDtype(torch.float32, scale=True)
                           ]))

    test = DataLoader(dataset=test_set,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=pin_memory
                      )
    return train_set, train_set, test_set
