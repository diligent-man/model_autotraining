import os, shutil

from tqdm import tqdm
from time import sleep

from typing import (
    List,
    Dict,
    Tuple,
    Any,
    Generator
)

from src_dev.utils import (
    Logger,
    LossManager,
    EarlyStopper,
    ModelManager,
    MetricManager,
    ConfigManager
)

from src_dev.open_src import (
    available_lr_scheduler,
    available_optimizers
)


import torch, torchinfo

from torch.utils.data import DataLoader

__all__ = ["Trainer"]


class Trainer:
    __config: ConfigManager
    __start_epoch: int = 1
    __logger: Logger = Logger()

    __train_loader: DataLoader
    __validation_loader: DataLoader

    __loss: torch.nn.Module
    __lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None

    __early_stopper: EarlyStopper = None
    __best_val_loss: float = None

    def __init__(self,
                 config: ConfigManager,
                 train_loader: DataLoader,
                 validation_loader: DataLoader
                 ):
        # Compulsory fields
        self.__config: ConfigManager = config

        self.__train_loader = train_loader
        self.__validation_loader = validation_loader

        self.__loss = LossManager(self.__config.LOSS_NAME, self.__config.LOSS_ARGS)

        self.__model = ModelManager(self.__config.MODEL_NAME,
                                    self.__config.MODEL_ARGS,
                                    self.__config.__dict__.get("MODEL_NEW_CLASSIFIER_NAME", None),
                                    self.__config.__dict__.get("MODEL_NEW_CLASSIFIER_ARGS", None),
                                    self.__config.DEVICE,
                                    self.__config.MODEL_PRETRAINED_WEIGHT
                                    ).model

        self.__optimizer = self.__init_optimizer(self.__config.OPTIMIZER_NAME,
                                                 self.__config.OPTIMIZER_ARGS,
                                                 self.__model.parameters()
                                                 )

        # Load checkpoint from local
        if self.__config.CHECKPOINT_LOAD:
            checkpoint = torch.load(f=os.path.join(self.__config.CHECKPOINT_PATH, self.__config.CHECKPOINT_RESUME_NAME), map_location=self.__config.DEVICE)
            self.__start_epoch = checkpoint["epoch"] + 1
            self.__model.load_state_dict(checkpoint["model_state_dict"])
            self.__optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            del checkpoint

        if self.__config.LR_SCHEDULER_APPLY:
            self.__lr_schedulers = self.__init_lr_scheduler(self.__config.LR_SCHEDULER_NAME,
                                                            self.__config.LR_SCHEDULER_ARGS,
                                                            self.__optimizer
                                                            )

        if self.__config.EARLY_STOPPING_APPLY:
            self.__best_val_loss = self.__get_best_val_loss()
            self.__early_stopper = EarlyStopper(self.__best_val_loss, **self.__config.EARLY_STOPPING_ARGS)

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

    # Setter & Getter
    @property
    def model(self):
        return self.__model

    def get_model_summary(self, depth=3, col_width=20, batch_size=1) -> torchinfo.summary:
        """
        if batch_dim is used, you only need to specify only input shape of img
        """
        input_size = self.__config.__dict__.get("DATA_INPUT_SHAPE", None)

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
                                 device=self.__config.DEVICE
                                 )

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
    def __train(self, epoch, data_loader, metrics):
        return self.__run_epoch("train", epoch, data_loader, metrics)


    def __eval(self, epoch, data_loader, metrics):
        run_epoch_result = self.__run_epoch("eval", epoch, data_loader, metrics)

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
                self.__lr_schedulers.step(epoch=epoch + index / len(data_loader)) ## Review lr scheduler mechanism

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
    def __init_optimizer(name: str,
                         args: Dict[str, Any],
                         model_paras: Generator
                         ) -> torch.optim.Optimizer:
        assert name in available_optimizers.keys(), "Your selected optimizer is unavailable."
        optimizer: torch.optim.Optimizer = available_optimizers[name](model_paras, **args)
        return optimizer

    @staticmethod
    def __init_lr_scheduler(name: str,
                            args: Dict[str, Any],
                            optimizer: torch.optim.Optimizer
                            ) -> torch.optim.lr_scheduler.LRScheduler:
        assert name in available_lr_scheduler.keys(), "Your selected lr scheduler is unavailable"
        return available_lr_scheduler[name](optimizer, **args)


    @staticmethod
    def __activate(pred_labels: torch.Tensor) -> None:
        if pred_labels.shape[1] == 1:
            # Binary class
            return torch.nn.functional.sigmoid(pred_labels).squeeze(dim=1)
        else:
            # Multiclass
            return torch.nn.functional.softmax(pred_labels, dim=1)
