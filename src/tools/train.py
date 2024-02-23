import os

from box import Box
from tqdm import tqdm
from time import time, sleep
from typing import List, Tuple, Dict
from src.utils.logger import Logger
from src.modelling.vgg import get_vgg_model
from src.modelling.resnet import get_resnet_model
from src.utils.early_stopping import EarlyStopper

import torch
import torcheval

from torch.nn.functional import sigmoid, softmax
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassF1Score, BinaryF1Score, MulticlassAccuracy, BinaryAccuracy
from torch.optim import Adam, AdamW, NAdam, RAdam, SparseAdam, Adadelta, Adagrad, Adamax, ASGD, RMSprop, Rprop, LBFGS, SGD
from torch.nn.modules import NLLLoss, NLLLoss2d, CTCLoss, KLDivLoss, GaussianNLLLoss, PoissonNLLLoss, L1Loss, MSELoss, HuberLoss, SmoothL1Loss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ConstantLR, LinearLR, ExponentialLR, PolynomialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ChainedScheduler, SequentialLR, ReduceLROnPlateau, OneCycleLR


class Trainer:
    """
    options: user-defined configs for entire training process
    best_acc:
    """
    __options: Box
    __train_log_path: str
    __eval_log_path: str
    __checkpoint_path: str
    __device: str

    __train_loader: DataLoader
    __validation_loader: DataLoader

    __early_stopper: EarlyStopper
    __logger: Logger

    __loss: torch.nn.Module
    __optimizer: torch.optim.Optimizer
    __lr_schedulers: torch.optim.lr_scheduler.LRScheduler
    __start_epoch: int
    __model: torch.nn.Module

    __best_val_loss: float


    def __init__(self, options: Box,
                 train_log_path: str, eval_log_path: str, checkpoint_path: str,
                 train_loader: DataLoader, validation_loader: DataLoader
                 ):
        self.__options: Box = options
        self.__train_log_path: str = train_log_path
        self.__eval_log_path: str = eval_log_path
        self.__checkpoint_path: str = checkpoint_path
        self.__device: str = "cuda" if self.__options.MISC.CUDA else "cpu"

        self.__train_loader: DataLoader = train_loader
        self.__validation_loader: DataLoader = validation_loader

        self.__early_stopper: EarlyStopper = EarlyStopper(**self.__options.SOLVER.EARLY_STOPPING)
        self.__logger: Logger = Logger()

        self.__loss = self.__init_loss()
        self.__start_epoch, self.__model, self.__optimizer = self.__init_model_optimizer_epoch()
        self.__lr_schedulers: torch.optim.lr_scheduler.LRScheduler = self.__init_lr_scheduler()

        self.__best_val_loss: float = self.__get_best_val_loss()


    @classmethod
    def __init_subclass__(cls):
        required_class_variables = [
            "__options", "__train_log_path", "__eval_log_path",  "__checkpoint_path", "__train_loader", "__val_loader"
        ]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(
                    f'Class {cls} lacks required `{var}` class attribute'
                )


    # Setter & Getter
    @property
    def model(self):
        return self.__model

    # Public methods
    def train(self, sleep_time: int = None) -> None:
        print("Start training model ...")

        for epoch in range(self.__start_epoch, self.__start_epoch + self.__options.EPOCH.EPOCHS):
            print("Epoch:", epoch)

            for phase, dataset_loader, log_path in zip(("train", "eval"), (self.__train_loader, self.__validation_loader), (self.__train_log_path, self.__eval_log_path)):
                # Preliminary setups
                self.__model.train() if phase == "train" else self.__model.eval()
                metrics = self.__init_metrics() if phase == "eval" else None

                # Epoch running
                run_epoch_result: Dict = self.__run_epoch(phase=phase, dataset_loader=dataset_loader, metrics=metrics)

                # Logging
                self.__logger.write(file=log_path, log_info={**{"epoch": epoch}, **run_epoch_result})

                if phase == "eval":
                    # Save checkpoint
                    if self.__options.CHECKPOINT.SAVE:
                        self.__save_checkpoint(epoch=epoch, val_loss=run_epoch_result["loss"],
                                               save_all=self.__options.CHECKPOINT.SAVE_ALL,
                                               obj={"epoch": epoch, "val_loss": run_epoch_result["loss"],
                                                    "model_state_dict": self.__model.state_dict(),
                                                    "optimizer_state_dict": self.__optimizer.state_dict()
                                                    }
                                               )

                    # Early stopping checking
                    if self.__options.MISC.APPLY_EARLY_STOPPING:
                        if self.__early_stopper.check(val_loss=run_epoch_result["loss"]):
                            exit()

                # Stop program in the meantime
                if sleep_time is not None:
                    sleep(sleep_time)
        return None


    # Private methods
    def __run_epoch(self, phase: str, dataset_loader: DataLoader, metrics: List[torcheval.metrics.Metric] = None) -> Dict:
        """
        phase: "train" || "eval"
        dataset_loader: train_loader || val_loader
        metrics: only available in eval phase

        Notes: loss of last iter is taken as loss of that epoch
        """
        num_class = self.__options.SOLVER.MODEL.ARGS.num_classes

        # Epoch training
        for index, batch in tqdm(enumerate(dataset_loader), total=len(dataset_loader), colour="cyan", desc=phase.capitalize()):
            imgs, labels = batch[0].type(torch.FloatTensor).to(self.__device), batch[1].type(torch.FloatTensor).to(self.__device)

            # reset gradients prior to forward pass
            self.__optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                # forward pass
                pred_labels = self.__model(imgs)
                pred_labels = sigmoid(pred_labels) if num_class == 1 else softmax(pred_labels)

                # NCHW shaoe
                if pred_labels.shape[1] == 1:
                    pred_labels = pred_labels.squeeze(1)

                # Update loss
                loss = self.__loss(pred_labels, labels)

                # backprop + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    self.__optimizer.step()

                # Update metrics only if eval phase
                if metrics is not None:
                    _ = [metric.update(pred_labels, labels) for metric in metrics]

        if metrics is not None:
            metrics_name = self.__options.METRICS.NAME_LIST
            metric_val = [metric.compute().item() for metric in metrics]
            training_result = {**{"loss": loss.item()},
                    **{metric_name: value for metric_name, value in zip(metrics_name, metric_val)}}
        else:
            training_result = {"loss": loss.item()}
        return training_result


    def __save_checkpoint(self, epoch: int, val_loss: float, obj: dict, save_all: bool = False) -> None:
        """
        save_all:
            True: save all trained epoch
            False: save only last and the best trained epoch
        Best_epoch is still saved in either save_all is True or False
        """
        save_name = os.path.join(self.__checkpoint_path, f"epoch_{epoch}.pt")
        torch.save(obj=obj, f=save_name)

        # Save best checkpoint
        if val_loss < self.__best_val_loss:
            save_name = os.path.join(self.__checkpoint_path, f"best_checkpoint.pt")
            torch.save(obj=obj, f=save_name)

            # Update best accuracy
            self.__best_val_loss = val_loss

        if not save_all and epoch - 1 > 0:
            # Remove previous epoch
            os.remove(os.path.join(self.__checkpoint_path, f"epoch_{epoch - 1}.pt"))
        return None


    def __get_best_val_loss(self) -> float:
        if "best_checkpoint.pt" in self.__checkpoint_path:
            return torch.load(f=os.path.join(self.__checkpoint_path, "best_checkpoint.pt"))["val_loss"]
        else:
            return 1e9


    def __init_model_optimizer_epoch(self) -> Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
        def init_optimizer(name: str, model_paras, state_dict: Dict = None, **kwargs) -> torch.optim.Optimizer:
            available_optimizers = {
                "Adam": Adam, "AdamW": AdamW, "NAdam": NAdam, "Adadelta": Adadelta, "Adagrad": Adagrad, "Adamax": Adamax,
                "RAdam": RAdam, "SparseAdam": SparseAdam, "RMSprop": RMSprop, "Rprop": Rprop, "ASGD": ASGD, "LBFGS": LBFGS, "SGD": SGD
            }
            assert name in available_optimizers.keys(), "Your selected optimizer is unavailable."

            # init optimizer
            optim: torch.optim.Optimizer = available_optimizers[name](model_paras, **kwargs)

            if state_dict is not None:
                optim.load_state_dict(state_dict)
            return optim

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


    def __init_lr_scheduler(self):
        available_lr_scheduler = {
            "LambdaLR": LambdaLR, "MultiplicativeLR": MultiplicativeLR, "StepLR": StepLR, "MultiStepLR": MultiStepLR, "ConstantLR": ConstantLR,
            "LinearLR": LinearLR, "ExponentialLR": ExponentialLR, "PolynomialLR": PolynomialLR, "CosineAnnealingLR": CosineAnnealingLR,
            "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts, "ChainedScheduler": ChainedScheduler, "SequentialLR": SequentialLR,
            "ReduceLROnPlateau": ReduceLROnPlateau, "OneCycleLR": OneCycleLR
        }
        lr_scheduler_name = self.__options.SOLVER.LR_SCHEDULER.NAME

        assert lr_scheduler_name in available_lr_scheduler.keys(), "Your selected lr scheduler is unavailable"
        return available_lr_scheduler[lr_scheduler_name](self.__optimizer, **self.__options.SOLVER.LR_SCHEDULER.ARGS)


    def __init_metrics(self) -> List[torcheval.metrics.Metric]:
        available_metrics = {
            "BinaryAccuracy": BinaryAccuracy,
            "BinaryF1Score": BinaryF1Score,

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
