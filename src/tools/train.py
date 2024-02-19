import os

from box import Box
from tqdm import tqdm
from time import time, sleep
from typing import List, Tuple
from src.utils.logger import Logger
from src.modelling.vgg import get_vgg_model
from src.modelling.resnet import get_resnet_model
from src.utils.early_stopping import EarlyStopper

import torch

from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassF1Score, MulticlassAccuracy
from torch.optim import (Adam, AdamW, NAdam, RAdam, SparseAdam, Adadelta, Adagrad, Adamax, ASGD, RMSprop, Rprop, LBFGS, SGD)
from torch.nn.modules import (NLLLoss, NLLLoss2d, CTCLoss, KLDivLoss, GaussianNLLLoss, PoissonNLLLoss, L1Loss, MSELoss, HuberLoss, SmoothL1Loss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss)


class Trainer:
    __options: Box
    __log_path: str
    __checkpoint_path: str
    __early_stopper: EarlyStopper
    __train_loss: torch.nn.Module
    __metrics: List
    __start_epoch: int
    __model: torch.nn.Module
    __optimizer: torch.optim.Optimizer
    __best_acc: float

    def __init__(self, options: Box, log_path: str, checkpoint_path: str):
        self.__options = options
        self.__log_path = log_path
        self.__checkpoint_path = checkpoint_path

        self.__early_stopper = EarlyStopper(**self.__options.SOLVER.EARLY_STOPPING)
        self.__train_loss = self.__init_loss()
        self.__metrics = self.__init_metrics()
        self.__start_epoch, self.__model, self.__optimizer = self.__init_model_optimizer_epoch()

        if not self.__options.CHECKPOINT.SAVE_ALL:
            self.__best_acc: float = self.__get_best_acc()

    # Setter & Getter
    @property
    def model(self):
        return self.__model

    # Public methods
    def train(self, train_set: DataLoader, validation_set: DataLoader, sleep_time: int = None) -> None:
        print("Start training model ...")
        self.__model.train()

        train_loss = None
        logger = Logger(log_path=self.__log_path, )

        # Start training
        for epoch in range(self.__start_epoch, self.__start_epoch + self.__options.EPOCH.EPOCHS):
            print("Epoch:", epoch)
            start_time = time()

            for index, batch in tqdm(enumerate(train_set), total=len(train_set), colour="cyan", desc="Training"):
                imgs, ground_truths = batch[0].type(torch.FloatTensor), batch[1]

                # Pass to predefined device
                if self.__options.MISC.CUDA:
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
            logger.write(writing_mode="a",
                         **{"epoch": epoch, "time per epoch": time() - start_time,
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
    def __validate(self, validation_set: DataLoader) -> List:
        self.__model.eval()
        val_loss = total_samples = 0
        val_metrics = self.__init_metrics()

        with torch.no_grad():
            for index, batch in tqdm(enumerate(validation_set), total=len(validation_set), desc="Validating"):
                imgs, ground_truths = batch[0].type(torch.FloatTensor), batch[1]

                if self.__options.MISC.CUDA:
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

    def __save_checkpoint(self, epoch: int, train_acc: float, obj: dict, save_all: bool = False) -> None:
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

    def __get_best_acc(self) -> float:
        if "best_checkpoint.pt" in self.__checkpoint_path:
            return torch.load(f=os.path.join(self.__checkpoint_path, "best_checkpoint.pt"))["train_acc"]
        else:
            return 0.

    def __init_model_optimizer_epoch(self) -> Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
        def init_optimizer(name: str, model_paras, state_dict=None, **kwargs) -> torch.optim.Optimizer:
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

        def init_model(cuda: bool, pretrained: bool, base: str, name: str, state_dict: dict, **kwargs) -> torch.nn.Module:
            available_bases = {
                "vgg": get_vgg_model,
                "resnet": get_resnet_model
            }
            assert base in available_bases.keys(), "Your selected base is unavailable"
            return available_bases[base](cuda, name, pretrained, state_dict, **kwargs)


        model_state_dict = None
        optimizer_state_dict = None
        start_epoch = self.__options.EPOCH.START

        if self.__options.CHECKPOINT.LOAD:
            map_location = "cuda" if self.__options.MISC.CUDA else "cpu"
            checkpoint = torch.load(f=os.path.join(self.__checkpoint_path, self.__options.CHECKPOINT.RESUME_NAME),
                                    map_location=map_location)
            start_epoch = checkpoint["epoch"] + 1
            model_state_dict = checkpoint["model_state_dict"]
            optimizer_state_dict = checkpoint["optimizer_state_dict"]


        model: torch.nn.Module = init_model(cuda=self.__options.MISC.CUDA,
                                            pretrained=self.__options.SOLVER.MODEL.PRETRAINED,
                                            base=self.__options.SOLVER.MODEL.BASE,
                                            name=self.__options.SOLVER.MODEL.NAME,
                                            state_dict=model_state_dict,
                                            **self.__options.SOLVER.MODEL.ARGS)

        optimizer: torch.optim.Optimizer = init_optimizer(name=self.__options.SOLVER.OPTIMIZER.NAME,
                                                          model_paras=model.parameters(),
                                                          state_dict=optimizer_state_dict,
                                                          **self.__options.SOLVER.OPTIMIZER.ARGS)
        return start_epoch, model, optimizer

    def __init_metrics(self) -> List:
        available_metrics = {
            "MulticlassAccuracy": MulticlassAccuracy,
            "MulticlassF1Score": MulticlassF1Score
        }

        # check whether metrics available or not
        for metric in self.__options.METRICS.NAME_LIST:
            assert metric in available_metrics.keys(), "Your selected metric is unavailable"

        metrics = []
        for i in range(len(self.__options.METRICS.NAME_LIST)):
            metrics.append(available_metrics[self.__options.METRICS.NAME_LIST[i]](**self.__options.METRICS.ARGS[str(i)]))

        if self.__options.MISC.CUDA:
            metrics = [metric.to("cuda") for metric in metrics]
        return metrics


    def __init_loss(self):
        available_loss = {
            "NLLLoss": NLLLoss, "NLLLoss2d": NLLLoss2d,
            "CTCLoss": CTCLoss, "KLDivLoss": KLDivLoss,
            "GaussianNLLLoss": GaussianNLLLoss, "PoissonNLLLoss": PoissonNLLLoss,
            "CrossEntropyLoss": CrossEntropyLoss, "BCELoss": BCELoss, "BCEWithLogitsLoss": BCEWithLogitsLoss,
            "L1Loss": L1Loss, "MSELoss": MSELoss, "HuberLoss": HuberLoss, "SmoothL1Loss": SmoothL1Loss,
        }
        assert self.__options.SOLVER.LOSS.NAME in available_loss.keys(), "Your selected loss function is unavailable"
        return available_loss[self.__options.SOLVER.LOSS.NAME](** self.__options.SOLVER.LOSS.ARGS)
