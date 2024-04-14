import torch
import torchinfo

from typing import Dict, List, Any, Generator
from src.utils.ConfigManager import ConfigManager
from src.open_src import available_model, available_weight, available_layer


__all__ = ["ModelManager"]


class ModelManager:
    __num_classes: int
    __model: torch.nn.Module

    def __init__(self,
                 model_name: str,
                 model_args: Dict[str, Dict],
                 new_classifier_name: List[str] = None,
                 new_classifier_args: List[Dict[str, Any]] = None,
                 device: str = "cpu",
                 pretrained_weight: bool = False
                 ):
        self.__num_classes = model_args.pop("num_classes", 1000)
        self.__model = self.__init_model(model_name, model_args, new_classifier_name, new_classifier_args, pretrained_weight).to(device)
    ##################################################################################################################


    # Setter & Getter
    @property
    def model(self):
        return self.__model

    @property
    def num_classes(self):
        return self.__num_classes
    ###################################################################################################################


    # Public methods
    def get_summary(self,
                          input_size,
                          depth=3,
                          col_width=20,
                          batch_size=1,
                          device="cpu",
                          verbose=True
                          ) -> torchinfo.model_statistics.ModelStatistics:
        """
        if batch_dim is used, you only need to specify only input shape of img
        """
        # Input shape must be [B, C, H, W]
        if len(input_size) == 3:
            input_size = [batch_size, *input_size]

        if verbose:
            col_names = ("input_size", "output_size", "num_params", "mult_adds", "params_percent", "trainable")
        else:
            col_names = ("num_params", "params_percent", "trainable")

        return torchinfo.summary(model=self.__model,
                                 input_size=input_size,
                                 col_names=col_names,
                                 col_width=col_width,
                                 depth=depth,
                                 device=device
                                 )
    ###################################################################################################################


    # Private methods
    def __init_model(self,
                     model_name: str,
                     model_args: Dict[str, Dict],
                     new_classifier_name: List[str] = None,
                     new_classifier_args: Dict[str, Dict] = None,
                     pretrained_weight: bool = False
                     ):
        assert model_name in available_model.keys(), "Your selected model is unavailable"

        if pretrained_weight:
            """Use pretrained weight from ImageNet-1K from pytorch hub"""
            # TODO: New classifier for AUX_LOGITS is undone -> still not train googlelenet & inceptionv3
            print("Loading pretrained weight")
            model = available_model[model_name](weights=available_weight[model_name].DEFAULT, **model_args)
            print("Adapting new classifier")
            model = self.__adapt_classifier(model, new_classifier_name, new_classifier_args)
        else:
            model = available_model[model_name](**{"num_classes": self.__num_classes, **model_args})
            # if state_dict is not None:
            #     print("Loading pretrained model...")
            #     model.load_state_dict(state_dict)
            #     print("Finished.")
            # else:
            print("Init model's parameters...")
            for para in model.parameters():
                if para.dim() > 1:
                    torch.nn.init.xavier_uniform_(para)
        print("Init finished.")
        return model


    def __adapt_classifier(self,
                           model: torch.nn.Module,
                           new_classifier_name: List[str] = None,
                           new_classifier_args: List[Dict[str, Any]] = None
                           ) -> torch.nn.Module:
        out_features: int = self._get_out_features(model.modules())
        if self.__num_classes != out_features:
            # Case 1 (Default): Add activation, dropout, linear to the last model's layer
            if new_classifier_name is None:
                default_classifier = torch.nn.Sequential(
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(p=0.5),
                    torch.nn.Linear(in_features=out_features, out_features=num_classes)
                    )

                model = torch.nn.Sequential(model, default_classifier)
            # Case 2: Supersede entire last module with specified configs
            else:
                model = torch.nn.Sequential(
                    *list(model.children())[:-1],
                    self.__get_new_classifier(new_classifier_name, new_classifier_args)
                )
        return model

    @staticmethod
    def _get_out_features(modules: Generator) -> int:
        # Retrieve from the last module
        for module in list(modules)[::-1]:
            if isinstance(module, torch.nn.Conv2d):
                return module.out_channels
            elif isinstance(module, torch.nn.Linear):
                return module.out_features

    @staticmethod
    def __get_new_classifier(new_classifier_name: List[str],
                             new_classifier_args: List[Dict[str, Any]]
                             ) -> torch.nn.Sequential:
        for layer in new_classifier_name:
            assert layer in available_layer, "Your selected layer is unavailable"

        new_classifier = torch.nn.Sequential(*[
            available_layer[name](**arg) for name, arg in zip(new_classifier_name, new_classifier_args)
        ])
        return new_classifier


# Test case: implement later
def main() -> None:
    # Your code
    config = ConfigManager(path="/home/trong/Downloads/Local/Source/python/semester_6/face_attribute/configs/vgg.json")
    model_manager = ModelManager(model_name = config.MODEL_NAME,
                                 model_args = config.MODEL_ARGS,
                                 device = config.DEVICE,
                                 new_classifier_name=config.MODEL_NEW_CLASSIFIER_NAME,
                                 new_classifier_args=config.MODEL_NEW_CLASSIFIER_ARGS,
                                 pretrained_weight=config.MODEL_PRETRAINED_WEIGHT
                                 )
    print(type(model_manager.model.parameters()))
    # torchinfo.summary(model_manager.model, depth=2, input_size=(1, 3, 224, 224))
    return None

if __name__ == '__main__':
    main()