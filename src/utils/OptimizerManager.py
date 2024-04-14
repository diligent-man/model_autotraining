import torch
from src.open_src import available_optimizers
from typing import Dict, Any, Generator


class OptimizerManager:
    def __init__(self,
                 name: str,
                 args: Dict[str, Any],
                 model_paras: Generator
                 ) -> torch.optim.Optimizer:
        assert name in available_optimizers.keys(), "Your selected optimizer is unavailable."
        self.__optimizer: torch.optim.Optimizer = available_optimizers[name](model_paras, **args)

    @property
    def optimizer(self):
        return self.__optimizer
