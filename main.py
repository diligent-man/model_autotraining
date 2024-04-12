import argparse

from src.tools import Trainer
from src.utils import DataManager, ConfigManager


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

    trainer = Trainer(config, train_loader, val_loader)
    trainer.get_model_summary()
    trainer.train()
    return None


def test(config: ConfigManager, data_manager: DataManager) -> None:
    test_loader = data_manager.get_test_loader(
        dataloader_args=config.DATA_TRAIN_LOADER_ARGS
    )
    print(f"Test: {len(test_loader)}")
    # evaluate(options=options, checkpoint_path=checkpoint_path, log_path=log_path, test_loader=test_loader)
    return None


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="/home/trong/Downloads/Local/Source/python/semester_6/face_attribute/configs/alexnet_multiclass.json", type=str, help="Path to config file")
    args = args.parse_args()

    config = ConfigManager(path=args.config)

    data_manager = DataManager(
        seed=config.SEED,
        device=config.DEVICE,
        transform=config.DATA_TRANSFORM,
        target_transform=config.DATA_TARGET_TRANSFORM
    )

    train(config, data_manager)
    return None


if __name__ == '__main__':
    main()


