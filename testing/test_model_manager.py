import yaml
import torch

from src.Manager import ModelManager
from typing import List, Tuple, Dict, Any


def preprocessing_testcase() -> List[Tuple[str, Dict[str, Any]]]:
    file = "config.yaml"
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
        [3, 512, 512],
        [3, 1024, 1024]
    ]
    __vebose: bool = False
    __device: str = "cuda" if torch.cuda.is_available() else "cpu"
    @staticmethod
    def pytest_generate_tests(metafunc):
        fn_name = metafunc.function.__name__
        scenarios = preprocessing_testcase()
        if fn_name in scenarios:
            metafunc.parametrize(argnames=scenarios[fn_name][0], argvalues=scenarios[fn_name][1], scope="function")

    def test_init_with_only_model_name(self, model_name):
        model = ModelManager(model_name, device=self.__device, verbose=self.__vebose).model
        assert isinstance(model, torch.nn.Module)

        for shape in self.__shapes:
            input = torch.randn(size=[1, *shape], device=self.__device, dtype=torch.float32)
            output = model(input)
            assert output.shape == (1, 1000)
            assert output.dtype == torch.float



    #
    # def test_init_with_model_args(sefl, model_name, model_args):
    #     assert isinstance(model_name, str)
    #
    #
    # def test_init_with_pretrained_weight(sefl, model_name, pretrained_weight):
    #     assert isinstance(model_name, str)