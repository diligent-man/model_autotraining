import json

from datetime import datetime
from multipledispatch import dispatch


class Logger:
    def __init__(self, phase: str = "train"):
        """
        phase: "train" || "test"
        """
        self.__time = {f"{phase.capitalize()} at": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    def write(self,  file: str, log_info: dict, writing_mode: str = "a") -> None:
        with open(file=file, mode=writing_mode, encoding="UTF-8", errors="ignore") as f:
            f.write(json.dumps(dict(self.__time, **log_info), indent=4))
            f.write(",\n")

    def write(self,  file: str, log_info: str, writing_mode: str = "w") -> None:
        with open(file=file, mode=writing_mode, encoding="UTF-8", errors="ignore") as f:
            f.write(log_info)
