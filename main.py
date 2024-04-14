# TODO: Model graph: https://stackoverflow.com/questions/52468956/how-do-i-visualize-a-net-in-pytorch

import argparse

from src.tools import Trainer
from src.utils import DataManager, ConfigManager
from src.utils import ModelManager

def train(config: ConfigManager, data_manager: DataManager) -> None:
    train_loader = data_manager.get_dataloader(
        dataset=config.DATA_DATASET,
        dataset_args=config.DATA_TRAIN_DATASET_ARGS,
        dataloader=config.DATA_DATALOADER,
        dataloader_args=config.DATA_TRAIN_DATALOADER_ARGS
    )

    val_loader = data_manager.get_dataloader(
        dataset=config.DATA_DATASET,
        dataset_args=config.DATA_VAL_DATASET_ARGS,
        dataloader=config.DATA_DATALOADER,
        dataloader_args=config.DATA_TRAIN_DATALOADER_ARGS
    )
    print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")



    print(model)
    # trainer = Trainer(config, train_loader, val_loader)
    # trainer.get_model_summary()
    # trainer.train()
    return None


def test(config: ConfigManager, data_manager: DataManager) -> None:
    test_loader = data_manager.get_test_loader(
        dataloader_args=config.DATA_TRAIN_LOADER_ARGS
    )
    print(f"Test: {len(test_loader)}")
    # evaluate(options=options, checkpoint_path=checkpoint_path, log_path=log_path, test_loader=test_loader)
    return None


def main(args: argparse.ArgumentParser) -> None:
    config = ConfigManager(path=args.config)

    data_manager = DataManager(
        seed=config.SEED,
        device=config.DEVICE,
        transform=config.DATA_TRANSFORM,
        target_transform=config.DATA_TARGET_TRANSFORM
    )

    model_manager = ModelManager(config.MODEL_NAME,
                                 config.MODEL_ARGS,
                                 config.__dict__.get("MODEL_NEW_CLASSIFIER_NAME", None),
                                 config.__dict__.get("MODEL_NEW_CLASSIFIER_ARGS", None),
                                 config.DEVICE,
                                 config.MODEL_PRETRAINED_WEIGHT
                                 )

    if config.MODEL_GET_SUMMARY:
        model_manager.get_summary(input_size=config.DATA_INPUT_SHAPE, device=config.DEVICE)


    # train(config, data_manager, model_manager.model)
    return None


if __name__ == '__main__':
    default_config_path = "/home/trong/Downloads/Local/Source/python/semester_6/face_attribute/configs/alexnet_binary_class.json"

    args = argparse.ArgumentParser()
    args.add_argument("--config", default=default_config_path, type=str, help="Path to config file")

    args = args.parse_args()
    main(args)


