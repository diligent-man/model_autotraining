import torch
from typing import Dict, Any, Iterator
from src.open_src import available_optimizers


__all__ = ["OptimizerManager"]


class OptimizerManager:
    def __init__(self,
                 name: str,
                 args: Dict[str, Any],
                 model_paras: Iterator[torch.nn.Parameter]
                 ) -> None:
        assert name in available_optimizers.keys(), "Your selected optimizer is unavailable."
        self.__optimizer: torch.optim.Optimizer = available_optimizers[name](model_paras, **args)

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer
