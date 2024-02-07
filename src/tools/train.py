import os

from box import Box
from time import time, sleep
from tqdm import tqdm
from typing import List
from src.utils.logger import Logger
from src.modelling.vgg import get_vgg_model
from src.modelling.resnet import get_resnet_model
from src.utils.early_stopping import EarlyStopper

import torch

from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from torch.optim import Adam, AdamW, NAdam, RAdam, SparseAdam, Adadelta, Adagrad, Adamax, ASGD, RMSprop, Rprop, LBFGS, SGD


class Trainer:
    def __init__(self, options: Box, log_path: str, checkpoint_path: str):
        self.__options: Box = options
        self.__log_path: str = log_path
        self.__checkpoint_path: str = checkpoint_path

        self.__metrics: List = self.__init_metrics()
        self.__early_stopper = EarlyStopper(**self.__options.EARLY_STOPPING)
        self.__train_loss: torch.nn = torch.nn.CrossEntropyLoss()

        if self.__options.CHECKPOINT.LOAD:
            map_location = "cuda" if self.__options.DEVICE.CUDA else "cpu"
            self.__checkpoint = torch.load(
                f=os.path.join(self.__checkpoint_path, self.__options.CHECKPOINT.RESUME_NAME),
                map_location=map_location)

            self.__model: torch.nn.Module = self.__init_model(cuda=self.__options.DEVICE.CUDA,
                                                              pretrained=self.__options.MODEL.PRETRAINED,
                                                              name=self.__options.MODEL.NAME,
                                                              state_dict=self.__checkpoint["model_state_dict"],
                                                              **self.__options.NN)

            self.__optimizer: torch.optim = self.__init_optimizer(name=self.__options.OPTIMIZER.NAME,
                                                                  model_paras=self.__model.parameters(),
                                                                  state_dict=self.__checkpoint["optimizer_state_dict"],
                                                                  **self.__options.OPTIMIZER.ARGS)
        else:
            self.__model: torch.nn.Module = self.__init_model(cuda=self.__options.DEVICE.CUDA,
                                                              pretrained=self.__options.MODEL.PRETRAINED,
                                                              name=self.__options.MODEL.NAME,
                                                              **self.__options.NN)
            self.__optimizer: torch.optim = self.__init_optimizer(name=self.__options.OPTIMIZER.NAME,
                                                                  model_paras=self.__model.parameters(),
                                                                  **self.__options.OPTIMIZER.ARGS)

        if not self.__options.CHECKPOINT.SAVE_ALL:
            self.__best_acc: float = self.__get_best_acc()


    # Setter & Getter
    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value

    # Public methods
    def train(self, train_set: DataLoader, validation_set: DataLoader, sleep_time: int = None) -> None:
        print("Start training model ...")
        self.__model.train()

        train_loss = None
        logger = Logger(log_path=self.__log_path)

        # Start training
        start_epoch = self.__checkpoint["epoch"] + 1 if self.__options.CHECKPOINT.LOAD else self.__options.EPOCH.START
        for epoch in range(start_epoch, start_epoch + self.__options.EPOCH.EPOCHS):
            print("Epoch:", epoch)
            start_time = time()

            for index, batch in tqdm(enumerate(train_set), total=len(train_set), colour="cyan", desc="Training"):
                imgs, ground_truths = batch[0].type(torch.FloatTensor), batch[1]

                # Pass to predefined device
                if self.__options.DEVICE.CUDA:
                    imgs = imgs.to("cuda")
                    ground_truths = ground_truths.to("cuda")

                # forward pass
                predictions = self.__model(imgs)
                train_loss = self.__train_loss(predictions, ground_truths)

                # backprop
                self.__optimizer.zero_grad()
                train_loss.backward()

                # updating weights
                self.__optimizer.step()

                # update metrics
                for metric in self.__metrics:
                    metric.update(predictions, ground_truths)

            # Training metrics
            train_acc, train_f1 = [metric.compute().item() for metric in self.__metrics]

            # Validate model
            val_loss, val_acc, val_f1 = self.__validate(validation_set)

            # Check early stopping cond
            if self.__options.MISC.APPLY_EARLY_STOPPING:
                if self.__early_stopper.early_stop(val_loss):
                    break

            # Logging
            logger.write(**{"epoch": epoch, "time per epoch": time() - start_time,
                            "train_loss": train_loss.item(), "train_acc": train_acc, "train_f1": train_f1,
                            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1
                            }
                         )

            # Save checkpoint
            if self.__options.CHECKPOINT.SAVE:
                self.__save_checkpoint(epoch=epoch, train_acc=train_acc,
                                       save_all=self.__options.CHECKPOINT.SAVE_ALL,
                                       obj={"epoch": epoch, "train_acc": train_acc,
                                            "model_state_dict": self.__model.state_dict(),
                                            "optimizer_state_dict": self.__optimizer.state_dict()
                                            }
                                       )
            # Reset metrics
            for metric in self.__metrics:
                metric.reset()

            # Stop in short time
            if sleep_time is not None:
                sleep(sleep_time)
        return None

    # Private methods
    def __init_metrics(self) -> List:
        metrics = [
            MulticlassAccuracy(num_classes=self.__options.NN.NUM_CLASSES),
            MulticlassF1Score(num_classes=self.__options.NN.NUM_CLASSES)
        ]

        if self.__options.DEVICE.CUDA:
            metrics = [metric.to("cuda") for metric in metrics]
        return metrics

    def __validate(self, validation_set: DataLoader) -> List:
        self.__model.eval()
        val_metrics = self.__init_metrics()
        val_loss = total_samples = 0

        with torch.no_grad():
            for index, batch in tqdm(enumerate(validation_set), total=len(validation_set), desc="Validating"):
                imgs, ground_truths = batch[0].type(torch.FloatTensor), batch[1]

                if self.__options.DEVICE.CUDA:
                    imgs = imgs.to("cuda")
                    ground_truths = ground_truths.to("cuda")

                # forward pass
                predictions = self.__model(imgs)
                val_loss += torch.nn.CrossEntropyLoss()(predictions, ground_truths) * imgs.size(0)
                total_samples += imgs.size(0)

                # update metrics
                for metric in val_metrics:
                    metric.update(predictions, ground_truths)

        val_loss /= total_samples
        return [val_loss.item()] + [metric.compute().item() for metric in val_metrics]

    def __get_best_acc(self) -> float:
        if "best_checkpoint.pt" in self.__checkpoint_path:
            return torch.load(f=os.path.join(self.__checkpoint_path, "best_checkpoint.pt"))["train_acc"]
        else:
            return 0.

    def __save_checkpoint(self, epoch: int, train_acc: float, obj: dict, save_all: bool = False, ) -> None:
        """
        save_all:
            True: save all trained epoch
            False: save only last and the best trained epoch
        """
        save_name = os.path.join(self.__checkpoint_path, f"epoch_{epoch}.pt")
        torch.save(obj=obj, f=save_name)

        if not save_all and epoch - 1 > 0:
            # Remove previous epoch
            os.remove(os.path.join(self.__checkpoint_path, f"epoch_{epoch - 1}.pt"))

            # Save best checkpoint
            if train_acc > self.__best_acc:
                save_name = os.path.join(self.__checkpoint_path, f"best_checkpoint.pt")
                torch.save(obj=obj, f=save_name)

                # Update best accuracy
                self.__best_acc = train_acc
        return None

    @staticmethod
    def __init_model(cuda: bool, pretrained: bool, name: str, state_dict: dict, **kwargs) -> torch.nn.Module:
        available_models = {
            "vgg": get_vgg_model,
            "resnet": get_resnet_model
        }
        selected_model = None

        for model in available_models.keys():
            if model in name:
                selected_model = available_models[model](cuda, pretrained, name, state_dict, **kwargs)

        assert selected_model is not None, "Your selected model is unavailable"
        return selected_model

    @staticmethod
    def __init_optimizer(name: str, model_paras, state_dict=None, **kwargs) -> torch.optim:
        available_optimizers = {
            "Adam": Adam,
            "AdamW": AdamW,
            "NAdam": NAdam,
            "Adadelta": Adadelta,
            "Adagrad": Adagrad,
            "Adamax": Adamax,
            "RAdam": RAdam,
            "SparseAdam": SparseAdam,
            "RMSprop": RMSprop,
            "Rprop": Rprop,
            "ASGD": ASGD,
            "LBFGS": LBFGS,
            "SGD": SGD
        }
        assert name in available_optimizers.keys(), "Your selected optimizer is unavailable."

        # init optimizer
        optimizer = available_optimizers[name](model_paras, **kwargs)

        if state_dict is not None:
            optimizer.load_state_dict(state_dict)
        return optimizer
