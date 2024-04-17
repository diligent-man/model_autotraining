import torch
from src.tools import Trainer
from src.Manager import DataManager, ConfigManager


def train(config: ConfigManager,
          model: torch.nn.Module,
          optimizer:torch.optim.Optimizer,
          data_manager: DataManager
          ) -> None:
    train_loader = data_manager.get_dataloader(
        config.DATA_DATASET,
        config.DATA_TRAIN_DATASET_ARGS,
        config.DATA_DATALOADER,
        config.DATA_TRAIN_DATALOADER_ARGS
    )

    val_loader = data_manager.get_dataloader(
        config.DATA_DATASET,
        config.DATA_VAL_DATASET_ARGS,
        config.DATA_DATALOADER,
        config.DATA_TRAIN_DATALOADER_ARGS
    )
    print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")

    trainer = Trainer(config, model, optimizer, train_loader, val_loader)
    trainer.train()
    return None


# def test(config: ConfigManager, data_manager: DataManager) -> None:
#     test_loader = data_manager.get_test_loader(
#         dataloader_args=config.DATA_TRAIN_LOADER_ARGS
#     )
#     print(f"Test: {len(test_loader)}")
#     # evaluate(options=options, checkpoint_path=checkpoint_path, log_path=log_path, test_loader=test_loader)
#     return None
