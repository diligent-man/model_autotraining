import torch
from typing import List, Dict, Union, Any
from torch.utils.tensorboard import SummaryWriter


__all__ = ['TensorboardManager']


class TensorboardManager:
    __summary_writer: SummaryWriter

    def __init__(self,
                 log_dir: str):
        self.__summary_writer = SummaryWriter(log_dir=log_dir)

    @property
    def summary_writer(self) -> SummaryWriter:
        return self.__summary_writer


    def add_graph(self,
                  model: torch.nn.Module,
                  input_shape: List[int],
                  device: str) -> None:
        self.__summary_writer.add_graph(
            model=model,
            input_to_model=torch.zeros(size=(1, *input_shape), device=device)
        )
        return None


    def add_scalar(self, tag: str, scalar_value: Union[float, int], global_step: int) -> None:
        self.__summary_writer.add_scalar(
            tag=tag,
            scalar_value=scalar_value,
            global_step=global_step
        )
        return None


    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, Any], global_step: int) -> None:
        self.__summary_writer.add_scalars(main_tag=main_tag,
                                          tag_scalar_dict=tag_scalar_dict,
                                          global_step=global_step
                                          )
        return None