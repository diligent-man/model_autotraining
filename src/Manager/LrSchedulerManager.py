import torch
from typing import Dict, Any, Union, List
from src.open_src import available_lr_scheduler


__all__ = ["LrSchedulerManager"]


# TODO: cannot run with ChainedScheduler, LambdaLR, MultiplicativeLR, SequentialLR, ReduceLROnPlateau,
class LrSchedulerManager:
    __lr_scheduler: torch.optim.lr_scheduler.LRScheduler

    def __init__(self,
                 schedulers: List[str],
                 args: Union[Dict[str, Any], List[Dict]],
                 optimizer: torch.optim.Optimizer
                 ) -> torch.optim.lr_scheduler.LRScheduler:
        for scheduler in schedulers:
            assert scheduler in available_lr_scheduler.keys(), "Your selected lr scheduler is unavailable"

        if len(schedulers) == 1:
            self.__lr_scheduler = available_lr_scheduler[schedulers.pop()](optimizer, **args.pop())
        else:
            scheduler_lst = [available_lr_scheduler[schedulers[i]](optimizer, **args[i]) for i in range(1, len(schedulers))]
            self.__lr_scheduler = available_lr_scheduler[schedulers[0]](optimizer, scheduler_lst, **args[0])


    @property
    def lr_scheduler(self):
        return self.__lr_scheduler








