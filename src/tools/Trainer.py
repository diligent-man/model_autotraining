import gc
import os
import copy
import shutil
from tqdm import tqdm

from time import sleep
from typing import List, Dict, Tuple, Any
from src.utils import Logger, EarlyStopper
from src.Manager import LossManager, MetricManager, ConfigManager, LrSchedulerManager, TensorboardManager

import torch
from torch.utils.data import DataLoader


__all__ = ["Trainer"]


torch.set_float32_matmul_precision('high')


class Trainer:
    __config: ConfigManager
    __sleep_time = None
    __start_epoch: int = 1
    __logger: Logger = Logger()

    __loss: LossManager
    __metrics: MetricManager
    __model: torch.nn.Module
    __optimizer: torch.optim.Optimizer

    __train_loader: DataLoader
    __validation_loader: DataLoader

    __lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
    __early_stopper: EarlyStopper = None
    __best_val_loss: float = None
    __tensorboard: TensorboardManager = None

    def __init__(self,
                 config: ConfigManager,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader,
                 validation_loader: DataLoader
                 ):
        # Compulsory fields
        self.__config = config
        self.__sleep_time = self.__config.__dict__.get("SLEEP", 0)

        self.__loss = LossManager(self.__config.LOSS_NAME, self.__config.LOSS_ARGS)
        self.__metrics = MetricManager(self.__config.METRIC_NAME,
                                       self.__config.METRIC_ARGS,
                                       self.__config.DEVICE)
        self.__model = torch.compile(model) if self.__config.MODEL_COMPILE else model
        self.__optimizer = optimizer
        self.__train_loader = train_loader
        self.__validation_loader = validation_loader

        # Load checkpoint from local
        if self.__config.CHECKPOINT_LOAD:
            checkpoint = torch.load(f=os.path.join(self.__config.CHECKPOINT_PATH, self.__config.CHECKPOINT_RESUME_NAME), map_location=self.__config.DEVICE)
            self.__start_epoch = checkpoint["epoch"] + 1
            self.__model.load_state_dict(checkpoint["model_state_dict"])
            self.__optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            del checkpoint

        if self.__config.LR_SCHEDULER_APPLY:
            self.__lr_scheduler = LrSchedulerManager(self.__config.LR_SCHEDULER_NAME,
                                                     self.__config.LR_SCHEDULER_ARGS,
                                                     self.__optimizer).lr_scheduler

        if self.__config.EARLY_STOPPING_APPLY:
            self.__best_val_loss = self.__get_best_val_loss()
            self.__early_stopper = EarlyStopper(self.__best_val_loss, **self.__config.EARLY_STOPPING_ARGS)

        if self.__config.TENSORBOARD_APPLY:
            self.__tensorboard = TensorboardManager(self.__config.TENSORBOARD_PATH)

            if self.__config.TENSORBOARD_INSPECT_MODEL and not self.__config.MODEL_COMPILE:
                # Tensorboard not support compiled pytorch model
                self.__tensorboard.add_graph(self.__model, self.__config.DATA_INPUT_SHAPE, self.__config.DEVICE)
    ##############################3###################################################################################3

    # Class methods
    @classmethod
    def __init_subclass__(cls):
        """Check indispensable args when instantiate Trainer"""
        required_class_variables = [
            "__config"
        ]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise AttributeError(
                    f'Class {cls} lacks required `{var}` class attribute'
                )
    ##################################################################################################################

    # Public methods
    def train(self) -> None:
        """
        sleep_time: temporarily cease the training process
        compute_metric_in_train: compute metrics during training phase or not
        """
        print("Start training model ...")

        for epoch in range(self.__start_epoch, self.__start_epoch + self.__config.TRAINING_EPOCHS):
            print("Epoch:", epoch)

            for phase, data_loader in zip(("train", "eval"), (self.__train_loader, self.__validation_loader)):
                # Preliminary setup
                if phase == "train":
                    self.__model.train()
                    run_epoch_result = self.__train(epoch, data_loader, copy.deepcopy(self.__metrics)) if self.__config.METRIC_IN_TRAIN else \
                                       self.__train(epoch, data_loader)
                elif phase == "eval":
                    self.__model.eval()
                    run_epoch_result = self.__eval(epoch, data_loader, copy.deepcopy(self.__metrics))

                # Logging
                self.__logger.write(f"{self.__config.LOG_PATH}/{phase}.json", {**{"epoch": epoch}, **run_epoch_result})

            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()

            # Stop program in the meantime
            print("Sleeping...")
            sleep(self.__sleep_time)


        # Remove pretrained weights in TORCH_HOME if exists
        if self.__config.MODEL_PRETRAINED_WEIGHT is not None and self.__config.MODEL_REMOVE_PRETRAINED_WEIGHT:
            shutil.rmtree(os.path.join(torch.hub._get_torch_home(), "hub", "checkpoints"))
        print("Training finished")
        return None
    #################################################################################################################3

    # Private methods
    def __train(self,
                epoch: int,
                data_loader: torch.utils.data.DataLoader,
                metrics: MetricManager,
                phase="train"
                ) -> Dict[str, Any]:
        run_epoch_result: Dict[str, Any] = {**{"Lr": self.__lr_scheduler.get_last_lr().pop()},
                            **self.__run_epoch(phase, epoch, data_loader, metrics)
                            }

        # Add to tensorboad writer
        if self.__tensorboard:
            self.__tensorboard.add_scalar("Learning rate", run_epoch_result["Lr"], epoch)
            self.__tensorboard.add_scalars("Loss",{phase: run_epoch_result["loss"]}, epoch)

            if self.__config.METRIC_IN_TRAIN:
                tag_scalar_dict: Dict[str, Any] = {f"{phase.capitalize()}_{metric}": run_epoch_result[metric] for metric in self.__config.TENSORBOARD_TRACKING_METRIC}
                self.__tensorboard.add_scalars("Metric", tag_scalar_dict, epoch)
        return run_epoch_result


    def __eval(self,
               epoch: int,
               data_loader: torch.utils.data.DataLoader,
               metrics: MetricManager,
               phase="eval"
               ) -> Dict[str, Any]:
        run_epoch_result: Dict[str, Any] = self.__run_epoch(phase, epoch, data_loader, metrics)

        # Add to tensorboad writer
        if self.__tensorboard:
            self.__tensorboard.add_scalars("Loss", {phase: run_epoch_result["loss"]}, epoch)

            tag_scalar_dict = {f"{phase.capitalize()}_{metric}": run_epoch_result[metric] for metric in self.__config.TENSORBOARD_TRACKING_METRIC}
            self.__tensorboard.add_scalars("Metric", tag_scalar_dict, epoch)

        # Save checkpoint
        if self.__config.CHECKPOINT_SAVE:
            model_state_dict = self.__model.state_dict() if self.__config.CHECKPOINT_SAVE_WEIGHT_ONLY else self.__model

            obj = {"epoch": epoch, "val_loss": run_epoch_result["loss"],
                   "model_state_dict": model_state_dict,
                   "optimizer_state_dict": self.__optimizer.state_dict()
                   }

            # Include config to checkpoint or not
            if self.__config.CHECKPOINT_INCLUDE_CONFIG:
                obj["config"] = self.__config.__dict__

            # Save cpkt
            self.__save_checkpoint(epoch=epoch, val_loss=run_epoch_result["loss"],
                                   save_all=self.__config.CHECKPOINT_SAVE_ALL,
                                   obj=obj
                                   )

        # Early stopping checking
        if self.__config.EARLY_STOPPING_APPLY and self.__early_stopper.check(val_loss=run_epoch_result["loss"]):
            exit()
        return run_epoch_result


    def __run_epoch(self,
                    phase: str,
                    epoch: int,
                    data_loader: DataLoader,
                    metrics: MetricManager,
                    ) -> Dict[str, Any]:
        """
        phase: "train" || "eval"
        data_loader: train_loader || val_loader
        metrics: only available in eval phase

        Notes: loss of last iter is taken as loss of that epoch
        """
        total_loss = self.__run_epoch_dispatcher(phase, epoch, data_loader, metrics)

        if metrics:
            metrics.compute()
            run_epoch_result = {**{"loss": total_loss / len(data_loader)},
                                **{metric_name: value for metric_name, value in zip(metrics.name, metrics.get_result())}
                                }
        else:
            run_epoch_result = {"loss": total_loss / len(data_loader)}
        return run_epoch_result


    def __run_epoch_dispatcher(self, phase, epoch, data_loader, metrics):
        # Ref: https://wandb.ai/wandb_fc/tips/reports/How-to-Properly-Use-PyTorch-s-CosineAnnealingWarmRestarts-Scheduler--VmlldzoyMTA3MjM2
        if isinstance(self.__lr_scheduler, (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)):
            """
            Template: Update lr after each step, batch
            scheduler = ...

                for epoch in range(...):
                    for phase in ("train", "val"):
                        for i, sample in enumerate(dataloader):
                            # Forward Pass

                            if phase == "train":
                                # Compute Loss and Backprop
                                # Update Optimizer
                                # Update SCheduler
                                scheduler.step() # < ----- Update Learning Rate after train each batch

            In brevity:
                for epoch in range():
                    train(batch_running(... + scheduler.step()))
                    val(batch_running(...))
            """
            total_loss = _run_epoch_strategy_2(self.__loss, self.__model, metrics,
                                               self.__optimizer, self.__lr_scheduler,
                                               phase, epoch, self.__config.DEVICE,
                                               self.__config.DATA_NUM_CLASSES, data_loader)
        else:
            """
            Template: update lr after each epoch
                scheduler = ...

                for epoch in range(...):
                    for phase in ("train", "val"):
                        for i, sample in enumerate(dataloader):
                            # Forward Pass

                            if phase == "train":
                                # Compute Loss and Backprop
                                # Update Optimizer

                        if phase == "val":
                            scheduler.step() # < ----- Update Learning Rate after run validation

            In brevity:
                for epoch in range():
                    train(batch_running(...))
                    val(batch_running(...))
                    scheduler.step()
            """
            total_loss = _run_epoch_strategy_1(self.__loss, self.__model, metrics,
                                               self.__optimizer, self.__lr_scheduler,
                                               phase, epoch, self.__config.DEVICE,
                                               self.__config.DATA_NUM_CLASSES, data_loader)
        return total_loss


    def __get_best_val_loss(self) -> float:
        if "best_checkpoint.pt" in os.listdir(self.__config.CHECKPOINT_PATH):
            return torch.load(f=os.path.join(self.__config.CHECKPOINT_PATH, "best_checkpoint.pt"))["val_loss"]
        else:
            return float("inf")


    def __save_checkpoint(self, epoch: int, val_loss: float, obj: dict, save_all: bool = False) -> None:
        """
        save_all:
            True: save all trained epoch
            False: save only last and the best trained epoch
        Best_epoch is still saved in either save_all is True or False
        """
        save_name = os.path.join(self.__config.CHECKPOINT_PATH, f"epoch_{epoch}.pt")
        torch.save(obj=obj, f=save_name)

        # Save best checkpoint
        if val_loss < self.__best_val_loss:
            # remove previous best epoch
            for name in os.listdir(self.__config.CHECKPOINT_PATH):
                if name.startswith("best"):
                    filepath = os.path.join(self.__config.CHECKPOINT_PATH, name)
                    os.remove(filepath)
                    break

            save_name = os.path.join(self.__config.CHECKPOINT_PATH, f"best_checkpoint_epoch_{epoch}.pt")
            torch.save(obj=obj, f=save_name)

            # Update best accuracy
            self.__best_val_loss = val_loss

        if not save_all and epoch - 1 > 0:
            # Remove previous epoch
            os.remove(os.path.join(self.__config.CHECKPOINT_PATH, f"epoch_{epoch - 1}.pt"))
        return None
    #################################################################################################################################


def _forward_pass(imgs: torch.Tensor, labels: torch.Tensor, num_classes: int,
                  model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  metrics: MetricManager, loss: LossManager,
                  phase: str, device: str) -> torch.FloatTensor:
    """
    Computation task in forward pass:
    1. Pass through model
    2. Compute batch loss
    3. Update metrics

    Return:
        batch_loss
    """
    def _activate(pred_labels: torch.Tensor) -> None:
        if pred_labels.shape[1] == 1:
            # Binary class
            return torch.nn.functional.sigmoid(pred_labels).squeeze(dim=1)
        else:
            # Multiclass
            return torch.nn.functional.softmax(pred_labels, dim=1)

    imgs = imgs.to(device)

    labels = labels.type(torch.FloatTensor) if num_classes == 1 else labels.type(torch.LongTensor)
    labels = labels.to(device)

    # reset gradients prior to forward pass
    optimizer.zero_grad()

    with torch.set_grad_enabled(phase == "train"):
        # forward pass
        pred_labels = model(imgs)
        pred_labels = list(map(_activate, pred_labels)) if isinstance(pred_labels, Tuple) else _activate(pred_labels)

        # Compute loss
        batch_loss: torch.FloatTensor = loss.compute_batch_loss(pred_labels, labels)

        # Get pred_labels from main output
        if isinstance(pred_labels, List): pred_labels = pred_labels[0]

        # Update metrics only if eval phase or metric_in_train == True
        if metrics: metrics.update(pred_labels, labels)
    return batch_loss


def _run_epoch_strategy_1(loss: LossManager, model: torch.nn.Module,
                          metrics: MetricManager, optimizer: torch.optim.Optimizer,
                          lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                          phase: str, epoch: int, device: str, num_classes: int,
                          data_loader: torch.utils.data.DataLoader) -> float:
    total_loss = 0

    # Epoch training
    for index, batch in tqdm(enumerate(data_loader), total=len(data_loader), colour="cyan", desc=phase.capitalize()):
        # img: (batch_size, input_shape), labels: (batch_size, )
        imgs, labels = batch
        batch_loss = _forward_pass(imgs, labels, num_classes, model, optimizer, metrics, loss, phase, device)
        total_loss += batch_loss.item()  # Accumulate minibatch into total loss

        # backprop + update optimizer
        if phase == "train":
            batch_loss.backward()
            optimizer.step()

    # Update scheduler after each epoch
    if lr_scheduler and phase == "eval":
        lr_scheduler.step()
    return total_loss


def _run_epoch_strategy_2(loss: LossManager, model: torch.nn.Module,
                          metrics: MetricManager, optimizer: torch.optim.Optimizer,
                          lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                          phase: str, epoch: int, device: str, num_classes: int,
                          data_loader: torch.utils.data.DataLoader
                          ) -> float:
    total_loss = 0

    # Epoch training
    for index, batch in tqdm(enumerate(data_loader), total=len(data_loader), colour="cyan", desc=phase.capitalize()):
        # img: (batch_size, input_shape), labels: (batch_size, )
        imgs, labels = batch
        batch_loss = _forward_pass(imgs, labels, num_classes, model, optimizer, metrics, loss, phase, device)

        # Accumulate minibatch into total loss
        total_loss += batch_loss.item()

        # backprop + optimize only if in training phase
        if phase == "train":
            batch_loss.backward()
            optimizer.step()

            # Update scheduler after each batch
            if lr_scheduler:
                lr_scheduler.step(epoch=epoch + index / len(data_loader))
    return total_loss