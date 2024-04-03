from typing import Dict, Any, Union, List

import torch
from src.open_src import available_loss


class LossManager:
    __loss: torch.nn.Module
    __name: str

    def __init__(self, name: str, args: Dict[str, Any]) -> None:
        assert name in available_loss.keys(), "Your selected loss function is unavailable"
        self.__loss: torch.nn.Module = available_loss[name](**args)
        self.__name = name

    @property
    def name(self):
        return self.__name

    def compute_batch_loss(self,
                           inputs: Union[torch.Tensor, List[torch.Tensor]],
                           targets: torch.Tensor,
                           aux_logits_weight: float=0.3) -> torch.Tensor:
        # len > 1: inputs includes aux logits (GoogleLeNet, InceptionV3)
        if isinstance(inputs, List):
            batch_loss = [self.__loss(inputs[i], targets) for i in range(len(inputs))]
            batch_loss = batch_loss[0] + sum(batch_loss) * aux_logits_weight
        elif isinstance(inputs, torch.Tensor):
            batch_loss = self.__loss(inputs, targets)
        return batch_loss
