import torch
from typing import Dict, Any
from src.open_src import available_lr_scheduler


class LrSchedulerManager:
    def __init__(self,
                 name: str,
                 args: Dict[str, Any],
                 optimizer: torch.optim.Optimizer
                 ) -> torch.optim.lr_scheduler.LRScheduler:
        assert name in available_lr_scheduler.keys(), "Your selected lr scheduler is unavailable"
        self.__lr_scheduler = available_lr_scheduler[name](optimizer, **args)

    @property
    def lr_scheduler(self):
        return self.__lr_scheduler
