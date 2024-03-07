import re

from custom_dataSet import CustomImageDataset
from json import JSONDecoder, JSONDecodeError

import torch

from torch.utils.data import Subset, DataLoader


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


def preliminary_checking() -> None:
    print("Torch version", torch.__version__)
    print("Cuda:", torch.cuda.is_available())
    print("Default dtype:", torch.get_default_dtype())
    print("Default autocast dtype to GPU:", torch.get_autocast_gpu_dtype())
    print("Default autocast dtype in CPU:", torch.get_autocast_cpu_dtype())

    print("These metrics solely take effect when running with cpu")
    print("Num of threads:", torch.get_num_threads())
    print("Num of inter-operations:", torch.get_num_interop_threads())
    return None
