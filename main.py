import os, copy
from pprint import pp

from src.tools.Trainer import Trainer
from src.utils.ConfigManager import ConfigManager


def train(config_manager: ConfigManager) -> None:
    trainer = Trainer(config_manager=config_manager)
    trainer.get_model_summary()
    trainer.train()
    return None

#
# def test(option_path: str) -> None:
#     for dataset in (["celeb_A", "collected_v3", "collected_v4"]):
#         options = Box(commentjson.loads(open(file=option_path, mode="r").read()))
#         checkpoint_path = os.path.join(os.getcwd(), "checkpoints", options.MODEL.NAME, options.CHECKPOINT.NAME)
#
#         options.DATA.DATASET_NAME = dataset
#         log_path = os.path.join(os.getcwd(), "logs", options.MODEL.NAME, f"testing_log_{dataset}.json")
#
#         test_set = get_dataset(root=os.path.join(os.getcwd(), options.DATA.DATASET_NAME, "test"),
#                                transform=options.DATA.TRANSFORM,
#                                )
#
#         test_loader: DataLoader = get_test_loader(dataset=test_set,
#                                                   batch_size=options.DATA.BATCH_SIZE,
#                                                   cuda=options.MISC.CUDA,
#                                                   num_workers=options.DATA.NUM_WORKERS
#                                                   )
#         print(f"""Test batch: {len(test_loader)}""")
#
#         evaluate(options=options, checkpoint_path=checkpoint_path, log_path=log_path, test_loader=test_loader)
#     return None


def main() -> None:
    # generate_celeb_A_dataset()
    config_manager = ConfigManager(path=os.path.join(os.getcwd(), "configs", "vgg.json"))
    train(config_manager)
    # test(option_path=os.path  .join(os.getcwd(), "configs", "test_config.json"))
    return None


if __name__ == '__main__':
    main()


