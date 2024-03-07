# https://github.com/minar09/VGG16-PyTorch/blob/master/main.py#L117
# https://pytorch-ignite.ai/how-to-guides/11-load-checkpoint/
# https://blog.paperspace.com/vgg-from-scratch-pytorch/
# https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
import os

from time import time
from tqdm import tqdm
from logger import Logger
from vgg import get_model

import torch

from torch.optim import Adam
from torchsummary import summary
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


class Trainer:
    path_options: dict
    dataset_options: dict
    checkpoint_options: dict
    optimizer_options: dict
    metric_options: dict
    device_options: dict
    nn_options: dict
    epoch_options: dict

    def __init__(self, path_options: dict, dataset_options: dict, checkpoint_options: dict,
                 optimizer_options: dict, metric_options: dict, device_options: dict,
                 nn_options: dict, epoch_options: dict):
        self.__path_options: dict = path_options
        self.__dataset_options: dict = dataset_options
        self.__checkpoint_options: dict = checkpoint_options
        self.__optimizer_options: dict = optimizer_options
        self.__metric_options: dict = metric_options
        self.__device_options: dict = device_options
        self.__nn_options: dict = nn_options
        self.__epoch_options: dict = epoch_options

        self.__checkpoint = None
        if self.__checkpoint_options["load_checkpoint"]:
            self.__checkpoint = torch.load(f=self.__checkpoint_options["checkpoint_path"],
                                           map_location=self.__device_options["device"]
                                           )
        self.__model = self.__initModel()
        self.__optimizer = self.__initAdam()

    # Public methods
    def train(self, train_set: DataLoader, test_set: DataLoader) -> None:
        print("Training model")

        # Init
        self.__model.train()
        logger = Logger(log_path=self.__path_options["log_path"])

        epoch = self.__epoch_options["start_epoch"]

        if self.__checkpoint_options["load_checkpoint"]:
            epoch = self.__checkpoint["epoch"] + 1

        # Start training
        for epoch in range(epoch, epoch + self.__epoch_options["epochs"]):
            start_time = time()
            total_sample = 0  # for computing avg of metrics

            for index, batch in tqdm(enumerate(train_set), total=len(train_set), colour="cyan", desc="Training"):
                imgs, labels = batch[0].type(torch.FloatTensor) / 255, batch[1]

                # Pass to predefined device
                if self.__device_options["cuda"]:
                    imgs = imgs.to(self.__device_options["device"])
                    labels = labels.to(self.__device_options["device"])

                # clear gradients of all vars (weights, biases) before performing backprop
                self.__optimizer.zero_grad()

                # Inference
                predictions = self.__model(imgs)

                # compute loss & backprop
                loss = CrossEntropyLoss()(predictions, labels)
                loss.backward()

                # Adjust optimizer paras
                self.__optimizer.step()

                # Accumulate metrics after each iteration
                self.__computeAccumulativeMetrics(predictions=predictions, labels=labels, flag="train")

                # for key in self.__metric_options["metrics"]:
                #     print(key, self.__metric_options["train_metrics"][key])

            # Take mean of metrics
            for key in self.__metric_options["metrics"]:
                self.__metric_options["train_metrics"][key] = 100 * self.__metric_options["train_metrics"][key] / total_sample

            # Evaluate model
            self.__eval(test_set)

            # Logging
            logger.write_to_file(**{"epoch": epoch,
                                    "time per epoch": time() - start_time,
                                    "loss": loss.item(),

                                    "train_accuracy": self.__metric_options["train_metrics"]["accuracy"],
                                    "train_recall": self.__metric_options["train_metrics"]["recall"],
                                    "train_precision": self.__metric_options["train_metrics"]["precision"],

                                    "eval_accuracy": self.__metric_options["eval_metrics"]["accuracy"],
                                    "eval_recall": self.__metric_options["eval_metrics"]["recall"],
                                    "eval_precision": self.__metric_options["eval_metrics"]["precision"]
                                    }
                                 )
            # Reset metrics
            for key1 in ("train", "eval"):
                for key2 in self.__metric_options[f"{key1}_metrics"].keys():
                    self.__metric_options[f"{key1}_metrics"][key2] = .0

            # Save model
            if self.__checkpoint_options["save_checkpoint"]:
                torch.save(obj={"epoch": epoch,
                                "model_state_dict": self.__model.state_dict(),
                                "adam_state_dict": self.__optimizer.state_dict()
                                }, f=os.path.join(self.__path_options["model_path"], f"epoch_{epoch}.pt")
                           )

            # for key in self.__metric_options["metrics"]:
            #     print(key, self.__metric_options["train_metrics"][key])
            # print(total_sample)
            # print()
            # print()
            # print()
        return None

    def getModel(self) -> summary:
        return summary(input_size=(3, self.__dataset_options["img_size"], self.__dataset_options["img_size"]),
                       model=self.__model, device=self.__device_options["device"]
                       )

    def getOptimizer(self):
        return self.__optimizer.state_dict()

    # Private methods
    def __eval(self, test_set) -> None:
        total_sample = 0

        with torch.no_grad():
            for index, batch in tqdm(enumerate(test_set), total=len(test_set), desc="Evaluating"):
                imgs, labels = batch[0].type(torch.FloatTensor) / 255, batch[1]

                if self.__device_options["cuda"]:
                    imgs = imgs.to(self.__device_options["device"])
                    labels = labels.to(self.__device_options["device"])

                # run the model on the test set to predict labels
                predictions = self.__model(imgs)

                # the label with the highest energy will be our prediction
                total_sample += labels.size(0)
                self.__computeAccumulativeMetrics(predictions=predictions, labels=labels, flag="eval")

        # Take mean
        for key in self.__metric_options["eval_metrics"].keys():
            self.__metric_options["eval_metrics"][key] = 100 * self.__metric_options["eval_metrics"][key] / total_sample
        return None

    def __computeAccumulativeMetrics(self, predictions: torch.Tensor, labels: torch.Tensor, flag: str) -> None:
        # compute & add up to current figure
        # print(predictions)
        _, predictions = torch.max(predictions.data, 1)
        # print(predictions)

        if flag == "train":
            for key in self.__metric_options["metrics"]:
                self.__metric_options["train_metrics"][key] += self.__metric_options["metric_funcs"][key](input=predictions, target=labels, threshold=self.__metric_options["threshold"])
        elif flag == "eval":
            for key in self.__metric_options["metrics"]:
                self.__metric_options["eval_metrics"][key] += self.__metric_options["metric_funcs"][key](input=predictions, target=labels, threshold=self.__metric_options["threshold"])
        return None

    def __initModel(self):
        if self.__checkpoint:
            model = get_model(cuda=self.__device_options["cuda"], device=self.__device_options["device"],
                              dropout=self.__nn_options["dropout"],
                              model_state_dict=self.__checkpoint["model_state_dict"]
                              )
        else:
            model = get_model(cuda=self.__device_options["cuda"], device=self.__device_options["device"],
                              dropout=self.__nn_options["dropout"]
                              )
        return model

    def __initAdam(self):
        adam = Adam(params=self.__model.parameters(),
                    lr=self.__optimizer_options["lr"],
                    betas=self.__optimizer_options["betas"],
                    eps=self.__optimizer_options["eps"],
                    weight_decay=self.__optimizer_options["weight_decay"],
                    amsgrad=self.__optimizer_options["amsgrad"]
                    )

        if self.__checkpoint:
            adam.load_state_dict(self.__checkpoint["adam_state_dict"])
        return adam