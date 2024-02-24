import json

from typing import Dict
from datetime import datetime


class Logger:
    def __init__(self, phase: str = "train"):
        """
        phase: "train" || "test"
        """
        self.__time = {f"{phase.capitalize()} at": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}


    def write(self,  file: str, log_info: Dict, writing_mode: str = "a"):
        # https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
        with open(file=file, mode=writing_mode, encoding="UTF-8", errors="ignore") as f:
            # for k in kwargs.keys():
            #     if isinstance(kwargs[k], torch.Tensor):
            #         kwargs[k] = float(kwargs[k])

            # kwargs = {print(v) for (k, v) in kwargs.items() if isinstance(v, tuple)}
            # print(kwargs)

            f.write(json.dumps(dict(self.__time, **log_info), indent=4))
            f.write(",\n")