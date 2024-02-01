import os

import torchmetrics
from box import Box
from time import time
from tqdm import tqdm
from typing import Tuple
from src.utils.logger import Logger
from src.modelling.vgg import get_model

import torch

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy


class Trainer:
    def __init__(self, options,
                 log_path=os.path.join(os.getcwd(), "logs", "training_log.json"),
                 checkpoint_path=os.path.join(os.getcwd(), "checkpoints")
                 ):
        self.__options: Box = options
        self.__log_path: str = log_path
        self.__checkpoint_path: str = checkpoint_path
        self.__best_acc: float = self.__get_best_acc()
        self.__checkpoint = None

        if self.__options.CHECKPOINT.LOAD:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            self.__checkpoint = torch.load(
                f=os.path.join(self.__checkpoint_path, self.__options.CHECKPOINT.RESUME_NAME),
                map_location=map_location)

        self.__model: torch.nn.Module = self.__init_model()
        self.__optimizer: torch.optim = self.__init_optimizer()
        self.__loss: torch.nn = torch.nn.CrossEntropyLoss()

        self.__acc: torchmetrics.Metric = self.__init_acc()
        self.__f1: torchmetrics.Metric = self.__init_f1_score()

    # Public methods
    def train(self, train_set: DataLoader, validation_set: DataLoader, loss=None) -> None:
        print("Start training model ...")

        self.__model.train()

        logger = Logger(log_path=self.__log_path)

        # Start training
        start_epoch = self.__checkpoint["epoch"] + 1 if self.__options.CHECKPOINT.LOAD else self.__options.EPOCH.START
        for epoch in range(start_epoch, start_epoch + self.__options.EPOCH.EPOCHS):
            print("Epoch:", epoch)
            start_time = time()

            for index, batch in tqdm(enumerate(train_set), total=len(train_set), colour="cyan", desc="Training"):
                imgs, ground_truths = batch[0].type(torch.FloatTensor), batch[1]

                # Pass to predefined device
                if torch.cuda.is_available():
                    imgs = imgs.to("cuda")
                    ground_truths = ground_truths.to("cuda")

                # forward pass
                predictions = self.__model(imgs)
                loss = self.__loss(predictions, ground_truths)

                # backprop
                self.__optimizer.zero_grad()
                loss.backward()

                # updating weights
                self.__optimizer.step()

                # update metrics
                self.__acc.update(predictions, ground_truths)
                self.__f1.update(predictions, ground_truths)

            # Training metrics
            train_acc, train_f1 = self.__acc.compute().item(), self.__f1.compute().item()

            # Validate model
            val_acc, val_f1 = self.__validate(validation_set)

            # Logging
            logger.write(**{"epoch": epoch,
                            "time per epoch": time() - start_time,
                            "loss": loss.item(),
                            "train_acc": train_acc,
                            "train_f1": train_f1,
                            "val_acc": val_acc,
                            "val_f1": val_f1
                            }
                         )

            # Save checkpoint
            if self.__options.CHECKPOINT.SAVE:
                # Save model after each epoch
                torch.save(obj={"epoch": epoch,
                                "train_acc": train_acc,
                                "model_state_dict": self.__model.state_dict(),
                                "optimizer_state_dict": self.__optimizer.state_dict()
                                },
                           f=os.path.join(self.__checkpoint_path, f"epoch_{epoch}.pt")
                           )
                # remove checkpoint of previous epoch
                if epoch - 1 > 0:
                    os.remove(os.path.join(self.__checkpoint_path, f"epoch_{epoch - 1}.pt"))

                # Save best checkpoint
                if train_acc > self.__best_acc:
                    torch.save(obj={"epoch": epoch,
                                    "train_acc": train_acc,
                                    "model_state_dict": self.__model.state_dict(),
                                    "optimizer_state_dict": self.__optimizer.state_dict()
                                    },
                               f=os.path.join(self.__checkpoint_path, f"best_checkpoint.pt")
                               )

            # Reset metrics
            self.__acc.reset()
            self.__f1.reset()
        return None

    def get_model_summary(self):
        input_size = (
            self.__options.DATA.INPUT_SHAPE[2],
            self.__options.DATA.INPUT_SHAPE[0],
            self.__options.DATA.INPUT_SHAPE[1]
        )
        return summary(self.__model, input_size=input_size)

    # Private methods
    def __init_model(self):
        if self.__checkpoint is not None:
            model = get_model(dropout=self.__options.NN.DROPOUT,
                              num_classes=self.__options.NN.NUM_CLASSES,
                              model_state_dict=self.__checkpoint["model_state_dict"]
                              )
        else:
            model = get_model(dropout=self.__options.NN.DROPOUT, num_classes=self.__options.NN.NUM_CLASSES)
        return model

    def __init_optimizer(self):
        optimizer = Adam(params=self.__model.parameters(), **self.__options.OPTIMIZER)

        if self.__checkpoint is not None:
            optimizer.load_state_dict(self.__checkpoint["optimizer_state_dict"])
        return optimizer

    @staticmethod
    def __init_acc():
        acc = MulticlassAccuracy(num_classes=2)

        if torch.cuda.is_available():
            acc.to("cuda")
        return acc

    @staticmethod
    def __init_f1_score():
        f1_score = MulticlassF1Score(num_classes=2)

        if torch.cuda.is_available():
            f1_score.to("cuda")
        return f1_score

    def __validate(self, validation_set: DataLoader):
        self.__model.eval()
        val_f1 = self.__init_f1_score()
        val_acc = self.__init_acc()

        with torch.no_grad():
            for index, batch in tqdm(enumerate(validation_set), total=len(validation_set), desc="Validating"):
                imgs, ground_truths = batch[0].type(torch.FloatTensor), batch[1]

                if torch.cuda.is_available():
                    imgs = imgs.to("cuda")
                    ground_truths = ground_truths.to("cuda")

                # run the model on the test set to predict labels
                predictions = self.__model(imgs)

                # update metrics
                val_acc.update(predictions, ground_truths)
                val_f1.update(predictions, ground_truths)
        return val_acc.compute().item(), val_f1.compute().item()

    def __get_best_acc(self) -> float:
        if "best_checkpoint.pt" in self.__checkpoint_path:
            return torch.load(f=os.path.join(self.__checkpoint_path, "best_checkpoint.pt"))["train_acc"]
        else:
            return 0.
