import os

from src.open_src import (
    available_conv,
    available_loss,
    available_linear,
    available_pooling,
    available_dropout,
    available_flatten,
    available_activation,
    available_normalization,
    available_optimizers,
    available_lr_scheduler,
    available_dtype,
    available_metrics,
    available_weight,
    available_transform,
    available_interpolation,
    available_model
)
from typing import Tuple, Dict, List, Any, Generator
from src.utils.ConfigManager import ConfigManager


import torch
import torcheval

from torchvision.transforms import Compose
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset


__all__ = ["get_train_val_loader", "get_test_loader", "init_loss", "init_lr_scheduler",
           "init_metrics", "init_model", "init_model_optimizer_start_epoch"
           ]


def _get_transformation(transforms: Dict[str, Dict] = None) -> Compose:
    if transforms is not None:
        # Verify transformation
        for k in transforms.keys():
            assert k in available_transform.keys(), "Your selected transform method is unavailable"

            # Verify interpolation mode & replace str name to its corresponding func
            if k in ("Resize", "RandomRotation"):
                assert transforms[k]["interpolation"] in available_interpolation.keys(), "Your selected interpolation mode in unavailable"
                transforms[k]["interpolation"] = available_interpolation[transforms[k]["interpolation"]]

            # Verify dtype & replace str name to its corresponding func
            if k in ("ToDtype"):
                assert transforms[k]["dtype"] in available_dtype.keys(), "Your selected dtype in unavailable"
                transforms[k]["dtype"] = available_dtype[transforms[k]["dtype"]]

        compose: Compose = Compose([available_transform[k](**v) for k, v in transforms.items()])
    else:
        compose: Compose = Compose([])
    return compose


def _get_dataset(root: str,
                transform: Dict[str, Dict] = None,
                target_transform: Dict[str, Dict] = None
                ) -> Dataset:
    """
    Args:
        root: dataset dir
        transform: Dict of transformation name and its corresponding kwargs
        target_transform:                     //                            but for labels
    """
    return ImageFolder(root=root,
                       transform=_get_transformation(transforms=transform),
                       target_transform=_get_transformation(transforms=target_transform)
                       )


def get_train_val_loader(config_manager: ConfigManager, customDataloader=None) -> Tuple[DataLoader, DataLoader]:
    dataset = _get_dataset(root=os.path.join(config_manager.DATA_PATH, "train"),
                           transform=config_manager.DATA_TRANSFORM,
                           target_transform=config_manager.DATA_TARGET_TRANSFORM
                           )

    train_size = round(len(dataset) * config_manager.DATA_TRAIN_SIZE)
    pin_memory = True if config_manager.DEVICE == "cuda" else False

    train_set, validation_set = random_split(dataset=dataset,
                                             generator=torch.Generator().manual_seed(config_manager.SEED),
                                             lengths=[train_size, len(dataset) - train_size])
    if customDataloader is None:
        train_set = DataLoader(dataset=train_set,
                               batch_size=config_manager.BATCH_SIZE,
                               shuffle=True,
                               num_workers=config_manager.DATA_NUM_WORKERS,
                               pin_memory=pin_memory
                               )

        validation_set = DataLoader(dataset=validation_set,
                                    batch_size=config_manager.BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=config_manager.DATA_NUM_WORKERS,
                                    pin_memory=pin_memory
                                    )
    else:
        train_set = customDataloader(dataset=train_set,
                                     batch_size=config_manager.BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=config_manager.DATA_NUM_WORKERS,
                                     pin_memory=pin_memory
                                     )

        validation_set = customDataloader(dataset=validation_set,
                                          batch_size=config_manager.BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=config_manager.DATA_NUM_WORKERS,
                                          pin_memory=pin_memory
                                          )
    return train_set, validation_set


def get_test_loader(dataset: Dataset,
                    batch_size: int = 32,
                    device: str = "cpu",
                    num_workers=1) -> DataLoader:
    # Use page-locked or not
    pin_memory = True if device == "cuda" else False
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=pin_memory
                      )
##########################################################################################################################


def init_loss(name: str, args: Dict[str, Any]) -> torch.nn.Module:
    assert name in available_loss.keys(), "Your selected loss function is unavailable"
    loss: torch.nn.Module = available_loss[name](**args)
    return loss


def init_lr_scheduler(name: str, args: Dict, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    assert name in available_lr_scheduler.keys(), "Your selected lr scheduler is unavailable"
    return available_lr_scheduler[name](optimizer, **args)


def init_metrics(name_lst: List[str], args: Dict, device: str) -> List[torcheval.metrics.Metric]:
    # check whether metrics available or not
    for metric in name_lst:
        assert metric in available_metrics.keys(), "Your selected metric is unavailable"

    metrics: List[torcheval.metrics.Metric] = []
    for i in range(len(name_lst)):
        metrics.append(available_metrics[name_lst[i]](**args[str(i)]))

    metrics = [metric.to(device) for metric in metrics]
    return metrics


def _adapt_classifier(model: torch.nn.Module,
                      num_classes: int,
                      new_classifier_name: List[str] = None,
                      new_classifier_args: List[Dict[str, Any]] = None
                      ) -> torch.nn.Module:
    def _get_out_features(modules: Generator) -> int:
        # Retrieve from the last module
        for module in list(modules)[::-1]:
            if isinstance(module, torch.nn.Conv2d):
                return module.out_channels
            elif isinstance(module, torch.nn.Linear):
                return module.out_features

    def _get_new_classifier(new_classifier_name: List[str],
                            new_classifier_args: List[Dict[str, Any]]
                            ) -> torch.nn.Sequential:
        available_layer = {
            **available_conv,
            **available_linear,
            **available_dropout,
            **available_flatten,
            **available_pooling,
            **available_activation,
            **available_normalization,
        }

        for layer in new_classifier_name:
            assert layer in available_layer, "Your selected layer is unavailable"

        return torch.nn.Sequential(
            *[available_layer[name](**arg) for name, arg in zip(new_classifier_name, new_classifier_args)]
        )

    out_features: int = _get_out_features(modules=model.modules())
    if num_classes != out_features:
        # Case 1 (Default): Add activation, dropout, linear to the last model's layer
        if new_classifier_name is None:
            model = torch.nn.Sequential(
                # Existent model
                model,
                # Adapted classifier
                torch.nn.Sequential(
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(p=0.5),
                    torch.nn.Linear(in_features=out_features, out_features=num_classes)
                )
            )
        # Case 2: Supersede entire last module with specified configs
        else:
            model = torch.nn.Sequential(
                *list(model.children())[:-1],
                _get_new_classifier(new_classifier_name, new_classifier_args)
                )
    return model


def init_model(device: str,
               num_classes: int,
               model_name: str,
               model_args: Dict[str, Any],
               new_classifier_name: List[str] = None,
               new_classifier_args: List[Dict[str, Any]] = None,
               state_dict: dict = None,
               pretrained_weight: bool = False,
               ) -> torch.nn.Module:
    # TODO: New classifier for AUX_LOGITS is undone -> still not train googlelenet & inceptionv3
    assert model_name in available_model.keys(), "Your selected model is unavailable"

    if pretrained_weight:
        """
        Use pretrained weight from ImageNet-1K"""
        model = available_model[model_name](weights=available_weight[model_name].DEFAULT, **model_args)
        model = _adapt_classifier(model, num_classes, new_classifier_name, new_classifier_args)
    else:
        model = available_model[model_name](**model_args)

        if state_dict is not None:
            print("Loading pretrained model...")
            model.load_state_dict(state_dict)
            print("Finished.")
        else:
            print("Initializing parameters...")
            for para in model.parameters():
                if para.dim() > 1:
                    torch.nn.init.xavier_uniform_(para)
            print("Finished.")

    # Temp solution for Inception_v3 when using pretrained weight
    # https://discuss.pytorch.org/t/questions-about-auxillary-classifier-of-inceptionv3/25211
    # from pprint import pprint as pp
    # if model_name in ("inception_v3", "googlenet"):
    #     if kwargs.get("aux_logits", True):
    #         for layer in model.modules():
    #             # if isinstance(layer, InceptionAux):
    #             print(layer)
    #
    #         # print(list(model.children())[-5])
    #         # print(list(model.children()))
    #     else:
    #         model.aux_logits = False
    model = model.to(device)
    return model


def init_optimizer(name: str, model_paras, state_dict: Dict = None, **kwargs) -> torch.optim.Optimizer:

    assert name in available_optimizers.keys(), "Your selected optimizer is unavailable."

    # init optimizer
    optimizer: torch.optim.Optimizer = available_optimizers[name](model_paras, **kwargs)

    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
    return optimizer


def init_model_optimizer_start_epoch(device: str,
                                     num_classes: int,
                                     checkpoint_load: bool,
                                     checkpoint_path: str,
                                     resume_name: str,
                                     optimizer_name: str,
                                     optimizer_args: Dict,
                                     model_name: str,
                                     model_args: Dict[str, Any],
                                     new_classifier_name: List[str],
                                     new_classifier_args: List[Dict[str, Any]],
                                     pretrained_weight: bool = False
                                     ) -> Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
    model_state_dict = None
    optimizer_state_dict = None
    start_epoch = 1

    if checkpoint_load:
        checkpoint = torch.load(f=os.path.join(checkpoint_path, resume_name), map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        model_state_dict = checkpoint["model_state_dict"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]

    # In case of saving entire model
    if not isinstance(model_state_dict, torch.nn.Module):
        model: torch.nn.Module = init_model(device=device,
                                            num_classes=num_classes,
                                            state_dict=model_state_dict,
                                            pretrained_weight=pretrained_weight,
                                            model_name=model_name,
                                            model_args=model_args,
                                            new_classifier_name=new_classifier_name,
                                            new_classifier_args=new_classifier_args
                                            )
    else:
        model = model_state_dict

    optimizer: torch.optim.Optimizer = init_optimizer(name=optimizer_name,
                                                      model_paras=model.parameters(),
                                                      state_dict=optimizer_state_dict,
                                                      **optimizer_args
                                                      )
    return start_epoch, model, optimizer

