import os

from box import Box
from src.tools.train import Trainer
from src.tools.eval import Evaluator
from src.tools.visualization import training_visualization
from src.utils.utils import json_decoder, get_dataset, get_model_summary


def main() -> None:
    options = Box(next(iter(json_decoder(open("config.json").read()))))
    # Load dataset
    train_set, validation_set, test_set = get_dataset(root=os.path.join(os.getcwd(), options.DATA.DATASET_NAME),
                                                      input_size=options.DATA.INPUT_SHAPE[0],
                                                      train_size=options.DATA.TRAIN_SIZE,
                                                      batch_size=options.DATA.BATCH_SIZE,
                                                      seed=options.MISC.SEED, cuda=options.DEVICE.CUDA,
                                                      num_workers=options.DATA.NUM_WORKERS
                                                      )
    print(f"""Train batch: {len(train_set)}, Validation batch: {len(validation_set)}, Test batch: {len(test_set)}""")

    # Start training
    trainer = Trainer(options=options,
                      log_path=os.path.join(os.getcwd(), "logs", f"{options.MODEL.NAME}_training_log.json"),
                      checkpoint_path=os.path.join(os.getcwd(), "checkpoints", options.MODEL.NAME))
    trainer.train(train_set, validation_set)

    # evaluator = Evaluator(options)
    # evaluator.eval(test_set, "best_checkpoint.pt")


    # training_visualization(file_name="training_log.json", metrics_lst=["loss", "acc", "f1"], x_interval=2)
    return None


if __name__ == '__main__':
    main()
