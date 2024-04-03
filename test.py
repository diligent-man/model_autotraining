import os

from pprint import pprint as pp

import torch

from src.utils.ConfigManager import ConfigManager
from src.utils.MetricManager import MetricManager


def main() -> None:
    # Your code
    config_manager = ConfigManager(path=os.path.join(os.getcwd(), "configs", "vgg.json"))
    metric_manager = MetricManager(metrics=config_manager.METRICS_NAME, args=config_manager.METRICS_ARGS, device=config_manager.DEVICE)

    inputs = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    targets = torch.tensor([0, 1, 2])

    metric_manager.update(inputs=inputs, targets=targets)
    metric_manager.compute()
    print(metric_manager.get_result())

    # print(metric_manager.get_result())



    return None

if __name__ == '__main__':
    main()