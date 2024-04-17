# TODO: Model graph: https://stackoverflow.2com/questions/52468956/how-do-i-visualize-a-net-in-pytorch
import argparse
from src.utils.utils import train
from src.utils import ModelManager, ConfigManager, OptimizerManager, DataManager


def main(args: argparse.ArgumentParser) -> None:
    config = ConfigManager(path=args.config)

    model_manager = ModelManager(config.MODEL_NAME,
                                 config.MODEL_ARGS,
                                 config.__dict__.get("MODEL_NEW_CLASSIFIER_NAME", None),
                                 config.__dict__.get("MODEL_NEW_CLASSIFIER_ARGS", None),
                                 config.DEVICE,
                                 config.MODEL_PRETRAINED_WEIGHT
                                 )
    if config.MODEL_GET_SUMMARY: model_manager.get_summary(input_size=config.DATA_INPUT_SHAPE, device=config.DEVICE)

    optimizer_manager = OptimizerManager(config.OPTIMIZER_NAME, config.OPTIMIZER_ARGS, model_manager.model.parameters())
    data_manager = DataManager(config.SEED, config.DEVICE, config.DATA_TRANSFORM, config.DATA_TARGET_TRANSFORM)

    train(config,
          model_manager.model,
          optimizer_manager.optimizer,
          data_manager
          )
    return None

import os
if __name__ == '__main__':
    path = "/home/trong/Downloads/Local/Source/python/semester_6/face_attribute/configs/LrScheduler/CosineAnnealingWarmRestarts.json"

    default_config_path = path

    args = argparse.ArgumentParser()
    args.add_argument("--config", default=default_config_path, type=str, help="Path to config file")

    args = args.parse_args()
    main(args)
