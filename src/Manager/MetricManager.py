import torch
import torcheval

from typing import List, Dict, Union, Any
from collections.abc import Iterable
from multipledispatch import dispatch
from src.open_src import available_metrics


__all__ = ["MetricManager"]


@dispatch(torch.Tensor)
def _get_metric_result(computed_metric: torch.Tensor) -> Union[float, List[float]]:
    return computed_metric.item() if computed_metric.dim() == 1 and len(computed_metric) == 1 else computed_metric.detach().cpu().numpy().tolist()


@dispatch(Iterable)
def _get_metric_result(computed_metric: Iterable) -> List[float]:
    result = []
    for constituent in computed_metric:
        if isinstance(constituent, List):
            result.append(_get_metric_result(constituent))
        else:
            if constituent.dim() == 0:
                result.append(_get_metric_result(constituent))
            else:
                contituent_result = []
                for tensor in constituent:
                    contituent_result.append(_get_metric_result(tensor))
                result.append(contituent_result)
    return result


class MetricManager:
    """
        Args:
            metrics: list of metrics
            args: kwargs of specific metric
            device: place metrics on cpu or gpu
    """
    __metrics: List[torcheval.metrics.Metric]
    __name: List[str]

    def __init__(self, metrics: List[str], args: List[Dict[str, Any]], device: str = "cpu"):
        for metric in metrics:
            assert metric in available_metrics.keys(), "Your selected metric is unavailable"
        self.__metrics = [available_metrics[metrics[i]](**args[i]) for i in range(len(metrics))]
        self.__name = metrics

    @property
    def metrics(self):
        return self.__metrics

    @property
    def name(self):
        return self.__name

    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        for metric in self.__metrics:
            try:
                metric.update(inputs, targets)
            except:
                if targets.dtype == torch.float:
                    targets = targets.type(torch.int)
                    metric.update(inputs, targets)
                elif targets.dtype == torch.int:
                    targets = targets.type(torch.float)
                    metric.update(inputs, targets)

    def compute(self) -> None:
        self.__metrics = [metric.compute() for metric in self.__metrics]

    def get_result(self):
        return [_get_metric_result(metric) for metric in self.__metrics]
