import os, shutil

from tqdm import tqdm
from time import sleep
from typing import List, Dict, Tuple, Any
from src.utils import Logger, LossManager, EarlyStopper, MetricManager, ConfigManager, LrSchedulerManager

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

__all__ = ["Trainer"]


class Trainer:
    __config: ConfigManager
    __start_epoch: int = 1
    __logger: Logger = Logger()

    __loss: torch.nn.Module
    __model: torch.nn.Module
    __optimizer: torch.optim.Optimizer
    __train_loader: DataLoader
    __validation_loader: DataLoader
    __lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
    __early_stopper: EarlyStopper = None
    __best_val_loss: float = None
    __tensorboard: SummaryWriter = None


    def __init__(self,
                 config: ConfigManager,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader,
                 validation_loader: DataLoader
                 ):
        # Compulsory fields
        self.__config = config

        self.__loss = LossManager(self.__config.LOSS_NAME, self.__config.LOSS_ARGS)
        self.__model = model
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
                                                     self.__optimizer
                                                     ).lr_scheduler

        if self.__config.EARLY_STOPPING_APPLY:
            self.__best_val_loss = self.__get_best_val_loss()
            self.__early_stopper = EarlyStopper(self.__best_val_loss, **self.__config.EARLY_STOPPING_ARGS)

        if self.__config.TENSORBOARD_APPLY:
            self.__tensorboard = SummaryWriter(log_dir=self.__config.TENSORBOARD_PATH)

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
    def train(self, sleep_time: int = None) -> None:
        """
        sleep_time: temporarily cease the training process
        compute_metric_in_train: compute metrics during training phase or not
        """
        print("Start training model ...")

        for epoch in range(self.__start_epoch, self.__start_epoch + self.__config.TRAINING_EPOCHS):
            print("Epoch:", epoch)

            for phase, data_loader in zip(("train", "eval"), (self.__train_loader, self.__validation_loader)):
                # Preliminary setups
                if phase == "train":
                    self.__model.train()
                    metrics: MetricManager = MetricManager(self.__config.METRIC_NAME,
                                                           self.__config.METRIC_ARGS,
                                                           self.__config.DEVICE
                                                           ) if self.__config.__dict__.get("METRIC_IN_TRAIN", False) else None

                    run_epoch_result: Dict[str, Any] = self.__train(epoch, data_loader, metrics)

                elif phase == "eval":
                    self.__model.eval()
                    metrics: MetricManager = MetricManager(self.__config.METRIC_NAME,
                                                           self.__config.METRIC_ARGS,
                                                           self.__config.DEVICE
                                                           )
                    run_epoch_result: Dict[str, Any] = self.__eval(epoch, data_loader, metrics)




                # Logging
                self.__logger.write(f"{self.__config.LOG_PATH}/{phase}.json", {**{"epoch": epoch}, **run_epoch_result})

                # Stop program in the meantime
                if sleep_time is not None:
                    sleep(sleep_time)

        # Remove pretrained weights in TORCH_HOME if exists
        if self.__config.MODEL_PRETRAINED_WEIGHT is not None and self.__config.MODEL_REMOVE_PRETRAINED_WEIGHT:
            shutil.rmtree(os.path.join(torch.hub._get_torch_home(), "hub", "checkpoints"))
        print("Training finished")
        return None
    #################################################################################################################3


    # Private methods
    def __train(self, epoch, data_loader, metrics, phase = "train"):
        run_epoch_result = {**{"Lr": self.__lr_scheduler.get_lr().pop()},
                            **self.__run_epoch(phase, epoch, data_loader, metrics)
                            }

        # Add to tensorboad writer
        if self.__tensorboard:
            self.__tensorboard.add_scalar(tag="Learning rate",
                                          scalar_value=run_epoch_result["Lr"],
                                          global_step=epoch
                                          )
            self.__tensorboard.add_scalars(main_tag="Loss",
                                           tag_scalar_dict={phase: run_epoch_result["loss"]},
                                           global_step=epoch
                                           )

            if self.__config.METRIC_IN_TRAIN:
                tag_scalar_dict = {f"{phase.capitalize()}_{metric}": run_epoch_result[metric] for metric in self.__config.TENSORBOARD_TRACKING_METRIC}
                self.__tensorboard.add_scalars(main_tag="Metric",
                                               tag_scalar_dict=tag_scalar_dict,
                                               global_step=epoch
                                               )
        return run_epoch_result


    def __eval(self, epoch, data_loader, metrics, phase = "eval"):
        run_epoch_result = self.__run_epoch(phase, epoch, data_loader, metrics)

        # Add to tensorboad writer
        if self.__tensorboard:
            self.__tensorboard.add_scalars(main_tag="Loss",
                                           tag_scalar_dict={phase: run_epoch_result["loss"]},
                                           global_step=epoch
                                           )

            if self.__config.METRIC_IN_TRAIN:
                tag_scalar_dict = {f"{phase.capitalize()}_{metric}": run_epoch_result[metric] for metric in self.__config.TENSORBOARD_TRACKING_METRIC}
                self.__tensorboard.add_scalars(main_tag="Metric",
                                               tag_scalar_dict=tag_scalar_dict,
                                               global_step=epoch
                                               )

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
        if self.__config.EARLY_STOPPING_APPLY:
            if self.__early_stopper.check(val_loss=run_epoch_result["loss"]):
                exit()
        return run_epoch_result


    def __run_epoch(self,
                    phase: str,
                    epoch: int,
                    data_loader: DataLoader,
                    metrics: MetricManager = None
                    ) -> Dict:
        """
        phase: "train" || "eval"
        data_loader: train_loader || val_loader
        metrics: only available in eval phase

        Notes: loss of last iter is taken as loss of that epoch
        """
        # print(phase, epoch, data_loader, metrics)
        num_class = self.__config.DATA_NUM_CLASSES
        total_loss = 0

        # Epoch training
        for index, batch in tqdm(enumerate(data_loader), total=len(data_loader), colour="cyan", desc=phase.capitalize()):
            imgs = batch[0].to(self.__config.DEVICE)

            labels = batch[1].type(torch.FloatTensor) if num_class == 1 else batch[1].type(torch.LongTensor)
            labels = labels.to(self.__config.DEVICE)

            if self.__config.TENSORBOARD_INSPECT_MODEL:
                self.__tensorboard.add_graph(self.__model, imgs)

            # reset gradients prior to forward pass
            self.__optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                # forward pass
                pred_labels = self.__model(imgs)
                pred_labels = list(map(self.__activate, pred_labels)) if isinstance(pred_labels, Tuple) else self.__activate(pred_labels)

                # Compute loss
                batch_loss = self.__loss.compute_batch_loss(pred_labels, labels)

                # Get pred_labels from main output
                if isinstance(pred_labels, List):
                    pred_labels = pred_labels[0]

                # Update metrics only if eval phase
                if metrics is not None:
                    metrics.update(pred_labels, labels)

            # Accumulate minibatch into total loss
            total_loss += batch_loss.item()

            # backprop + optimize only if in training phase
            if phase == 'train':
                batch_loss.backward()
                self.__optimizer.step()
                self.__lr_scheduler.step(epoch=epoch + index / len(data_loader)) ## Review lr scheduler mechanism

        if metrics is not None:
            metrics.compute()
            run_epoch_result = {**{"loss": total_loss / len(data_loader)},
                                **{metric_name: value for metric_name, value in zip(metrics.name, metrics.get_result())}
                                }
        else:
            run_epoch_result = {"loss": total_loss / len(data_loader)}
        return run_epoch_result


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

    @staticmethod
    def __activate(pred_labels: torch.Tensor) -> None:
        if pred_labels.shape[1] == 1:
            # Binary class
            return torch.nn.functional.sigmoid(pred_labels).squeeze(dim=1)
        else:
            # Multiclass
            return torch.nn.functional.softmax(pred_labels, dim=1)
