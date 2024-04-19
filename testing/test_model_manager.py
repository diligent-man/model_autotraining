#Ref: https://stackoverflow.com/questions/66906406/parametrizing-multiple-tests-dynamically-in-python
import gc
import os
import yaml
import torch
import torchvision

from src.Manager import ModelManager
from typing import List, Tuple, Dict, Any


def preprocessing_testcase() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Desc in yaml will be ignored
    """
    file = os.path.join(os.getcwd(), "configs", "model_manager_config.yaml")
    with open(file=file, mode="r", encoding="UTF-8", errors="ignore") as f:
        testcase = yaml.safe_load(f.read())
    scenarios = {}

    for scenario in testcase["scenarios"]:
        scenario_name: str = scenario["name"]
        scenarios[scenario_name] = [scenario["argnames"], scenario["argvals"]]
    return scenarios


class TestModelManager:
    __shapes = [
        [3, 224, 224],
        [3, 299, 299],
        [3, 512, 512],
        [3, 1024, 1024]
    ]
    __num_classes = [1, 2, 1000]
    __scenarios = preprocessing_testcase()
    __vebose: bool = False
    __batch_size: int = 1
    __device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def pytest_generate_tests(metafunc):
        fn_name = metafunc.function.__name__
        scenarios = metafunc.cls._TestModelManager__scenarios

        if fn_name in scenarios:
            metafunc.parametrize(argnames=scenarios[fn_name][0], argvalues=scenarios[fn_name][1], scope="function")

    # def test_init_with_shape(self, model_name):
    #     model_manager = ModelManager(model_name, device=self.__device, verbose=self.__vebose)
    #     model = model_manager.model
    #     num_classes = model_manager.num_classes
    #     assert isinstance(model, torch.nn.Module)
    #     del model_manager
    #
    #     for shape in self.__shapes:
    #         input = torch.randn(size=[self.__batch_size, *shape], device=self.__device, dtype=torch.float32)
    #         # Move to cpu in case of CUDA out of mem
    #         try:
    #             output = model(input)
    #         except:
    #             try:
    #                 output = model.to("cpu")(input.to("cpu"))
    #             except:
    #                 print(f"{model_name} not accept shape {shape}")
    #                 continue
    #
    #         if self.__device == "cuda":
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #
    #         if model_name in ("inception_v3", "googlenet"):
    #             # ignore aux_logits from inception & googlenet
    #             assert output.logits.shape == (self.__batch_size, num_classes)
    #             assert output.logits.dtype == torch.float32
    #         else:
    #             assert output.shape == (self.__batch_size, num_classes)
    #             assert output.dtype == torch.float32
    # #     print("")
    #
    # def test_init_with_num_classes(self, model_name: str, input_shape: List[int]):
    #     input = torch.randn(size=[self.__batch_size, *input_shape], device=self.__device, dtype=torch.float32)
    #
    #     for num_class in self.__num_classes:
    #         model_manager = ModelManager(model_name, model_args={"num_classes": num_class}, device=self.__device, verbose=self.__vebose)
    #         model = model_manager.model
    #         assert isinstance(model, torch.nn.Module)
    #         del model_manager
    #
    #         # Move to cpu in case of CUDA out of mem
    #         try:
    #             output = model(input)
    #         except:
    #             try:
    #                 output = model.to("cpu")(input.to("cpu"))
    #             except:
    #                 continue
    #
    #         if self.__device == "cuda":
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #
    #         if model_name in ("inception_v3", "googlenet"):
    #             # ignore aux_logits from inception & googlenet
    #             assert output.logits.shape == (self.__batch_size, num_class)
    #             assert output.logits.dtype == torch.float32
    #         else:
    #             assert output.shape == (self.__batch_size, num_class)
    #             assert output.dtype == torch.float32
    #     print("")

    # def test_init_with_pretrained_weight_and_default_new_head(self, model_name, pretrained_weight):
    #     for num_class in self.__num_classes:
    #         model_manager = ModelManager(model_name, model_args={"num_classes": num_class}, pretrained_weight=pretrained_weight, verbose=self.__vebose)
    #         model = model_manager.model
    #         assert isinstance(model, torch.nn.Module)
    #         del model_manager
    #
    #         # if model_name in ("inception_v3", "googlenet"):
    #         #     print(model)
    #         #     # ignore aux_logits from inception & googlenet
    #         #     assert model.logits.shape == (self.__batch_size, num_class)
    #         #     assert model.logits.dtype == torch.float32
    #         # else:
    #         if num_class == 1000:
    #             # Keep model intact
    #             if model_name in ("squeezenet1_0", "squeezenet1_1"):
    #                 assert list(model.modules())[-3].out_channels == num_class
    #             else:
    #                 assert list(model.modules())[-1].out_features == num_class
    #         else:
    #             # Encapsulate model in Sequential module with default new head (ReLU -> Dropout -> Linear)
    #             assert model[-1].out_features == num_class
    #         # assert model[-1].out_features == num_class
    #     print("\n")
