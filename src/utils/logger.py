import json

from datetime import datetime


class Logger:
    __log_path: str
    __checkpoint: bool

    def __init__(self, log_path: str):
        self.__log_path = log_path
        self.__train_at = {"Train at": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    def write(self, writing_mode: str = "a", **kwargs: dict):
        # https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
        with open(file=self.__log_path, mode=writing_mode, encoding="UTF-8", errors="ignore") as f:
            # for k in kwargs.keys():
            #     if isinstance(kwargs[k], torch.Tensor):
            #         kwargs[k] = float(kwargs[k])

            # kwargs = {print(v) for (k, v) in kwargs.items() if isinstance(v, tuple)}
            # print(kwargs)

            f.write(json.dumps(dict(self.__train_at, **kwargs), indent=4))
            f.write("\n")