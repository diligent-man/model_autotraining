# TODO: Reimplement DataManager: not apply data augmentation on val set
# TODO: Change config transform & target transform into List[] and List[Dict] format\
#       instead of Dict[] as current cuz some operation can happene twice (e.g. resize)
# TODO: Implement eval strategy
# TODO: Implement inference process
# TODO: Refactor import of src (use relative import)
# TODO: Implement customized ImageFolder dataset that has capability of reading image as tensor not PIL as current
# TODO: Use eval() to extract arg instead of dict access by key
import gc
import torch
import argparse

from src.utils.utils import train
from src.Manager import ModelManager, ConfigManager, OptimizerManager, DataManager


def main(args: argparse.ArgumentParser) -> None:
    config = ConfigManager(path=args.config)

    model_manager = ModelManager(config.MODEL_NAME,
                                 config.__dict__.get("MODEL_ARGS", {}),
                                 config.__dict__.get("MODEL_NEW_CLASSIFIER_NAME", None),
                                 config.__dict__.get("MODEL_NEW_CLASSIFIER_ARGS", None),
                                 config.__dict__.get("MODEL_PRETRAINED_WEIGHT", False),
                                 config.__dict__.get("DEVICE", "cpu"),
                                 config.__dict__.get("VERBOSE", True),
                                 )

    if config.MODEL_GET_SUMMARY:
        model_manager.get_summary(
            config.DATA_INPUT_SHAPE,
            device=config.DEVICE
        )

    optimizer_manager: OptimizerManager = OptimizerManager(
        config.OPTIMIZER_NAME,
        config.OPTIMIZER_ARGS,
        model_manager.model.parameters()
    )

    data_manager: DataManager = DataManager(
        config.SEED,
        config.DEVICE,
        config.DATA_TRANSFORM,
        config.DATA_TARGET_TRANSFORM
    )

    train(config,
          model_manager.model,
          optimizer_manager.optimizer,
          data_manager
          )
    return None


if __name__ == '__main__':
    for name in ["swin_s.json", "swin_t.json", "swin_s_v2.json", "swin_t_v2.json"]:
        path = f"./configs/DSP391m/{name}"

        args = argparse.ArgumentParser()
        args.add_argument("--config", default=path, type=str, help="Path to config file")

        args = args.parse_args()
        main(args)

        torch.cuda.empty_cache()
        gc.collect()
