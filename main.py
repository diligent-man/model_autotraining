import os
from collections import Counter

from src.tools.Trainer import Trainer
from src.utils.DataManager import DataManager
from src.utils.ConfigManager import ConfigManager


def train(config: ConfigManager, data_manager: DataManager) -> None:
    train_loader, val_loader = data_manager.get_train_val_loader(
        train_size=config.DATA_TRAIN_SIZE,
        dataloader_args=config.DATA_TRAIN_LOADER_ARGS
    )
    print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")

    trainer = Trainer(config, train_loader, val_loader)
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


def main() -> None:
    # generate_celeb_A_dataset()
    config = ConfigManager(path=os.path.join(os.getcwd(), "configs", "vgg.json"))

    data_manager = DataManager(
        root=config.DATA_PATH,
        seed=config.SEED,
        device=config.DEVICE,
        transform=config.DATA_TRANSFORM,
        target_transform=config.DATA_TARGET_TRANSFORM
    )

    train(config, data_manager)
    return None


if __name__ == '__main__':
    main()


