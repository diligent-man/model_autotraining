import os, shutil

from tqdm import tqdm
from time import sleep
from typing import List, Dict, Tuple, Union
from src.utils.Logger import Logger
from src.utils.EarlyStopper import EarlyStopper
from src.utils.ConfigManager import ConfigManager
from src.utils.utils import get_train_val_loader
from src.utils.utils import init_loss, init_metrics, init_lr_scheduler, init_model_optimizer_start_epoch

import torch, torcheval, torchinfo
from torch.utils.data import DataLoader
from torchvision.models.googlenet import GoogLeNetOutputs
from torchvision.models.inception import InceptionOutputs


class Trainer:
    __config_manager: ConfigManager
    __train_loader: DataLoader
    __validation_loader: DataLoader

    __logger: Logger = Logger()
    __loss: torch.nn.Module
    __lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None

    __early_stopper: EarlyStopper = None
    __best_val_loss: float = None

    def __init__(self, config_manager: ConfigManager):
        # Compulsory fields
        self.__config_manager: ConfigManager = config_manager
        self.__train_loader, self.__validation_loader = get_train_val_loader(self.__config_manager)
        self.__loss = init_loss(self.__config_manager.LOSS_NAME, self.__config_manager.LOSS_ARGS)
        self.__start_epoch, self.__model, self.__optimizer = init_model_optimizer_start_epoch(device=self.__config_manager.DEVICE,
                                                                                              num_classes=self.__config_manager.DATA_NUM_CLASSES,
                                                                                              checkpoint_load=self.__config_manager.CHECKPOINT_LOAD,
                                                                                              checkpoint_path=self.__config_manager.CHECKPOINT_PATH,
                                                                                              resume_name=self.__config_manager.CHECKPOINT_RESUME_NAME,
                                                                                              optimizer_name=self.__config_manager.OPTIMIZER_NAME,
                                                                                              optimizer_args=self.__config_manager.OPTIMIZER_ARGS,
                                                                                              model_name=self.__config_manager.MODEL_NAME,
                                                                                              model_args=self.__config_manager.MODEL_ARGS,
                                                                                              model_classifier_name=self.__config_manager.__dict__.get("MODEL_CLASSIFIER_NAME", None),
                                                                                              model_classifier_args=self.__config_manager.__dict__.get("MODEL_CLASSIFIER_ARGS", None),
                                                                                              pretrained_weight=self.__config_manager.MODEL_PRETRAINED_WEIGHT
                                                                                              )

        if self.__config_manager.LR_SCHEDULER_APPLY:
            self.__lr_schedulers = init_lr_scheduler(self.__config_manager.LR_SCHEDULER_NAME, self.__config_manager.LR_SCHEDULER_ARGS, self.__optimizer)

        if self.__config_manager.EARLY_STOPPING_APPLY:
            self.__best_val_loss = self.__get_best_val_loss()
            self.__early_stopper = EarlyStopper(self.__best_val_loss, **self.__config_manager.EARLY_STOPPING_ARGS)
        #
        # from operator import add
        # def print_class_counts(data_loader_dict):
        #     counter_lst = [0] * 2
        #     for epoch in range(3):
        #         for phase, data_loader in data_loader_dict.items():
        #             print(f"Phase: {phase}")
        #             for i, (inputs, labels) in enumerate(data_loader):
        #                 class_counts = labels.bincount()
        #                 # print(f"Batch {i + 1}: {class_counts.tolist()}")
        #
        #                 if len(class_counts.tolist()) < 2:
        #                     counter_lst = list( map(add, counter_lst, class_counts.tolist() + [0]))
        #                 else:
        #                     counter_lst = list(map(add, counter_lst, class_counts.tolist()))
        #                     print(counter_lst)
        #             print()
        #             print()
        #
        # print_class_counts({
        #     "train": self.__train_loader
        # })

    # Class methods
    @classmethod
    def __init_subclass__(cls):
        """Check indispensable args when instantiate Trainer"""
        required_class_variables = [
            "__config_manager"
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


    def get_model_summary(self, depth=3, col_width=20, batch_size=1) -> torchinfo.summary:
        """
        if batch_dim is used, you only need to specify only input shape of img
        """
        input_size = self.__config_manager.__dict__.get("DATA_INPUT_SHAPE", None)

        # Input shape must be [B, C, H, W]
        if len(input_size) != 3:
            input_size = None

        if input_size is not None:
            input_size = (batch_size, *input_size)
            col_names = ("input_size", "output_size", "num_params", "mult_adds", "params_percent", "trainable")
        else:
            col_names = ("num_params", "params_percent", "trainable")
        return torchinfo.summary(model=self.__model,
                                 input_size=input_size,
                                 col_names=col_names,
                                 col_width=col_width,
                                 depth=depth,
                                 device=self.__config_manager.DEVICE)

    # Public methods
    def train(self, sleep_time: int = None, compute_metric_in_train: bool = False) -> None:
        """
        sleep_time: temporarily cease the training process
        compute_metric_in_train: compute metrics during training phase or not
        """
        print("Start training model ...")

        for epoch in range(self.__start_epoch, self.__start_epoch + self.__config_manager.TRAINING_EPOCHS):
            print("Epoch:", epoch)

            for phase, dataset_loader in zip(("train", "eval"), (self.__train_loader, self.__validation_loader)):
                # Preliminary setups
                self.__model.train() if phase == "train" else self.__model.eval()
                metrics: List[torcheval.metrics.Metric] = init_metrics(name_lst=self.__config_manager.METRICS_NAME,
                                                                       args=self.__config_manager.METRICS_ARGS,
                                                                       device=self.__config_manager.DEVICE) if compute_metric_in_train else None

                # Epoch running
                run_epoch_result: Dict = self.__run_epoch(phase=phase, epoch=epoch, dataset_loader=dataset_loader, metrics=metrics)

                # Logging
                self.__logger.write(f"{self.__config_manager.LOG_PATH}/{phase}.json", {**{"epoch": epoch}, **run_epoch_result})

                if phase == "eval":
                    # Save checkpoint
                    if self.__config_manager.CHECKPOINT_SAVE:
                        model_state_dict = self.__model.state_dict() if self.__config_manager.CHECKPOINT_SAVE_WEIGHT_ONLY else self.__model

                        obj = {"epoch": epoch, "val_loss": run_epoch_result["loss"],
                               "model_state_dict": model_state_dict,
                               "optimizer_state_dict": self.__optimizer.state_dict()
                               }

                        # Include config to checkpoint or not
                        if self.__config_manager.CHECKPOINT_INCLUDE_CONFIG:
                            obj["config"] = self.__config_manager.__dict__

                        # Save cpkt
                        self.__save_checkpoint(epoch=epoch, val_loss=run_epoch_result["loss"],
                                               save_all=self.__config_manager.CHECKPOINT_SAVE_ALL,
                                               obj=obj
                                               )

                    # Early stopping checking
                    if self.__config_manager.EARLY_STOPPING_APPLY:
                        if self.__early_stopper.check(val_loss=run_epoch_result["loss"]):
                            exit()

                # Stop program in the meantime
                if sleep_time is not None:
                    sleep(sleep_time)

        # Remove pretrained weights in TORCH_HOME if exists
        if self.__config_manager.MODEL_PRETRAINED_WEIGHT is not None and self.__config_manager.MODEL_REMOVE_PRETRAINED_WEIGHT:
            shutil.rmtree(os.path.join(torch.hub._get_torch_home(), "hub", "checkpoints"))
        print("Training finished")
        return None

    # Private methods
    def __compute_batch_loss(self, pred_labels: Union[List[torch.Tensor], torch.Tensor], labels: torch.Tensor):
        # len > 1: pred_labels includes aux logits (GoogleLeNet, InceptionV3)
        if isinstance(pred_labels, List):
            batch_loss = [self.__loss(pred_labels[i], labels) for i in range(len(pred_labels))]
            batch_loss = batch_loss[0] + sum(batch_loss) * self.__config_manager.__dict__.get("MODEL_AUX_LOGITS_WEIGHT",
                                                                                              0.3)
            return pred_labels[0], batch_loss

        elif isinstance(pred_labels, torch.Tensor):
            batch_loss = self.__loss(pred_labels, labels)
            return pred_labels, batch_loss

    def __get_best_val_loss(self) -> float:
        if "best_checkpoint.pt" in os.listdir(self.__config_manager.CHECKPOINT_PATH):
            return torch.load(f=os.path.join(self.__config_manager.CHECKPOINT_PATH, "best_checkpoint.pt"))["val_loss"]
        else:
            return float("inf")

    def __save_checkpoint(self, epoch: int, val_loss: float, obj: dict, save_all: bool = False) -> None:
        """
        save_all:
            True: save all trained epoch
            False: save only last and the best trained epoch
        Best_epoch is still saved in either save_all is True or False
        """
        save_name = os.path.join(self.__config_manager.CHECKPOINT_PATH, f"epoch_{epoch}.pt")
        torch.save(obj=obj, f=save_name)

        # Save best checkpoint
        if val_loss < self.__best_val_loss:
            # remove previous best epoch
            for name in os.listdir(self.__config_manager.CHECKPOINT_PATH):
                if name.startswith("best"):
                    filepath = os.path.join(self.__config_manager.CHECKPOINT_PATH, name)
                    os.remove(filepath)
                    break

            save_name = os.path.join(self.__config_manager.CHECKPOINT_PATH, f"best_checkpoint_epoch_{epoch}.pt")
            torch.save(obj=obj, f=save_name)

            # Update best accuracy
            self.__best_val_loss = val_loss

        if not save_all and epoch - 1 > 0:
            # Remove previous epoch
            os.remove(os.path.join(self.__config_manager.CHECKPOINT_PATH, f"epoch_{epoch - 1}.pt"))
        return None

    def __run_epoch(self, phase: str, epoch: int,
                    dataset_loader: DataLoader,
                    metrics: List[torcheval.metrics.Metric] = None
                    ) -> Dict:
        """
        phase: "train" || "eval"
        dataset_loader: train_loader || val_loader
        metrics: only available in eval phase

        Notes: loss of last iter is taken as loss of that epoch
        """
        num_class = self.__config_manager.DATA_NUM_CLASSES
        total_loss = 0

        # Epoch training
        for index, batch in tqdm(enumerate(dataset_loader), total=len(dataset_loader), colour="cyan", desc=phase.capitalize()):
            imgs = batch[0].to(self.__config_manager.DEVICE)

            labels = batch[1].type(torch.FloatTensor) if num_class == 1 else batch[1].type(torch.LongTensor)
            labels = labels.to(self.__config_manager.DEVICE)

            # reset gradients prior to forward pass
            self.__optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                # forward pass
                pred_labels = self.__model(imgs)
                pred_labels = list(map(self.__activate, pred_labels)) if isinstance(pred_labels, Tuple) else self.__activate(pred_labels)

                # Compute loss
                pred_labels, batch_loss = self.__compute_batch_loss(pred_labels, labels)

                # Update metrics only if eval phase
                if metrics is not None:
                    metrics = [metric.update(pred_labels, labels) for metric in metrics]

            # Accumulate minibatch into total loss
            total_loss += batch_loss.item()

            # backprop + optimize only if in training phase
            if phase == 'train':
                batch_loss.backward()
                self.__optimizer.step()
                # epoch=epoch + index / len(dataset_loader
                self.__lr_schedulers.step()

        if metrics is not None:
            metrics_name = self.__config_manager.METRICS_NAME

            for i in range(len(metrics_name)):
                metrics[i] = metrics[i].compute()
                # In case of metric return tensor
                if isinstance(metrics[i], torch.Tensor):
                    metrics[i] = metrics[i].item() if metrics[i].dim() == 1 and len(metrics[i]) == 1 else metrics[i].detach().cpu().numpy().tolist()

                # In case of metric return tuple
                elif isinstance(metrics[i], tuple):
                    metrics[i] = [ele.detach().cpu().numpy().tolist() for ele in metrics[i]]

            training_result = {**{"loss": total_loss / len(dataset_loader)},
                               **{metric_name: value for metric_name, value in zip(metrics_name, metrics)}
                               }
        else:
            training_result = {"loss": total_loss / len(dataset_loader)}
        return training_result

    # Static methods
    @staticmethod
    def __activate(pred_labels: torch.Tensor) -> None:
        if pred_labels.shape[1] == 1:
            # Binary class
            return torch.nn.functional.sigmoid(pred_labels).squeeze(dim=1)
        else:
            # Multiclass
            return torch.nn.functional.softmax(pred_labels, dim=1)