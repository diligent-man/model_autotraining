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


__all__ = ["init_model", "init_model_optimizer_start_epoch"]





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

