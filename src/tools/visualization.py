import os
import shutil
import matplotlib

from typing import List
from matplotlib import pyplot as plt
from src.utils.utils import json_decoder

matplotlib.use('TkAgg')
# print(matplotlib.get_backend())


def training_visualization(file_name: str, metrics_lst: List[str], x_interval=2):
    # remove existing dir
    if os.path.exists(os.path.join(os.getcwd(), "report")):
        shutil.rmtree(os.path.join(os.getcwd(), "report"))
        os.mkdir(os.path.join(os.getcwd(), "report"), mode=0x777)
    else:
        os.mkdir(os.path.join(os.getcwd(), "report"), mode=0x777)

    # Retrieve metrics from training log
    f = json_decoder(open(os.path.join(os.getcwd(), "logs", file_name)).read())
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

        ax.set_xticks(range(min(metrics_dict["epoch"]), max(metrics_dict["epoch"])+2, x_interval))

        ax.legend(["Train", "Validation"])
        fig.savefig(os.path.join(os.getcwd(), "report", f"{metrics_lst[i].title()}.jpg"))