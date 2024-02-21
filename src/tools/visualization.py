import os
import json
import matplotlib

from typing import List

import numpy as np
from matplotlib import pyplot as plt


matplotlib.use('TkAgg')
# print(matplotlib.get_backend())


def training_log_visualization(file_name: str, base_name: str, metrics_lst: List[str], x_interval=2, y_interval=0.1):
    report_path = os.path.join(os.getcwd(), "report", base_name)
    log_path = os.path.join(os.getcwd(), "logs", file_name)

    os.makedirs(report_path, mode=0x777, exist_ok=True)

    # Retrieve metrics from training log (ignore last line and last comma)
    f = json.loads("[" + open(log_path).read()[:-2] + "]")

    metrics_dict = {}

    for json_obj in f:
        for metric in metrics_lst:
            if f"train_{metric}" not in metrics_dict.keys():
                metrics_dict["epoch"] = {json_obj["epoch"]}
                metrics_dict[f"train_{metric}"] = [json_obj[f"train_{metric}"]]
                metrics_dict[f"val_{metric}"] = [json_obj[f"val_{metric}"]]
            else:
                metrics_dict["epoch"].add(json_obj["epoch"])
                metrics_dict[f"train_{metric}"].append(json_obj[f"train_{metric}"])
                metrics_dict[f"val_{metric}"].append(json_obj[f"val_{metric}"])

    # Visualize metrics
    for i in range(len(metrics_lst)):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_title(f"{metrics_lst[i].title()} Visualization")

        ax.plot(list(metrics_dict["epoch"]), metrics_dict[f"train_{metrics_lst[i]}"], label="a", color="red")
        ax.plot(list(metrics_dict["epoch"]), metrics_dict[f"val_{metrics_lst[i]}"], label="b", color="cyan")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metrics_lst[i].title())

        ax.set_xticks(range(0, max(metrics_dict["epoch"])+2, x_interval))
        ax.set_yticks([round(num, 1) for num in np.linspace(start=0, stop=1, num=10)])
        ax.legend(["Train", "Validation"])
        fig.savefig(os.path.join(report_path, f"{metrics_lst[i].title()}.jpg"))