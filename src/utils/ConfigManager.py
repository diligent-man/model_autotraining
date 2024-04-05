import os
import commentjson

from typing import Dict, Any

__all__ = ['ConfigManager']


class ConfigManager:
    def __init__(self, path: str):
        config = commentjson.loads(open(file=path, mode="r", encoding="UTF-8").read())
        self.__init_dynamic_field(config)
        self.__post_init()

    # Public methods
    def set(self, field: str, value: Any) -> None:
        self.__dict__[field] = value

    def get(self, field: str):
        return self.__dict__.get(field, None)

    # Private methods
    def __init_dynamic_field(self, config: Dict[str, Any], max_recursive_level: int = 0) -> None:
        """
        Create class dynamic fields of configs with one recursive level
        """
        if max_recursive_level < 1:
            for key in config.keys():
                if isinstance(config.get(key), Dict):
                    self.__init_dynamic_field(
                        {f"{key}_{sub_key}": config[key][sub_key] for sub_key in config[key].keys() if config.get(key)},
                        max_recursive_level + 1
                    )
                else:
                    self.__init_dynamic_field({f"{key}": config[key]}, max_recursive_level + 1)
        else:
            # Create class field
            for k, v in config.items():
                setattr(self, k, v)
        return None

    def __post_init(self):
        # Check dataset
        self.__check_dataset_format()

        # Set output path
        self.__set_output_path()
        # check paths existence
        # self.__check_output_path()

    def __check_dataset_format(self) -> None:
        """
        Things to check
            1/ Existence
            2/ Format

        Dataset must be as follows:
            root
                train
                    class_1
                    class_2
                    ...
                test
                    class_1
                    class_2
                    ...
        This format will be read by ImageFolder from Pytorch
        """
        # Check existence
        assert os.path.isdir(self.DATA_PATH), "Provided dataset path does not exist"

        # Check format
        for dataset in ("train", "test"):
            assert dataset in os.listdir(self.DATA_PATH), f"{dataset} set is not provided"
        return None

    def __set_output_path(self) -> None:
        """
        Check existence of checkpoint and log path
        If not exists, create dir as the following pattern:
            output/checkpoint/<MODEL_BASE>/<MODEL_NAME>
                 /log/<MODEL_BASE>/<MODEL_NAME>
        """
        output_path = self.__dict__.get("OUTPUT_PATH", os.path.join(os.getcwd(), "output"))

        for path in ("checkpoint", "log"):
            # Add path to class attr
            k = f"{path.upper()}_PATH"
            v = os.path.join(os.getcwd(), output_path, self.MODEL_BASE, self.MODEL_NAME, path)
            self.__dict__[k] = v

            # Create dir if not exists
            if not os.path.isdir(v):
                os.makedirs(v, 0o777, True)
                print(f"Dir for {k.lower()} {self.MODEL_NAME} is created.")
        return None

