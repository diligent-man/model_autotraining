import os

from box import Box
from src.tools.train import Trainer
from src.tools.eval import Evaluator
from src.tools.visualization import training_visualization
from src.utils.utils import json_decoder, get_dataset


def main() -> None:
    options = Box(next(iter(json_decoder(open("config.json").read()))))

    train_set, validation_set, test_set = get_dataset(root=os.path.join(os.getcwd(), options.DATA.DATASET_NAME),
                                                      img_size=options.DATA.INPUT_SHAPE[0],
                                                      train_size=options.DATA.TRAIN_SIZE,
                                                      batch_size=options.DATA.BATCH_SIZE,
                                                      seed=options.MISC.SEED, cuda=options.DEVICE.CUDA,
                                                      num_workers=options.DATA.NUM_WORKERS
                                                      )
    print(f"""Train batch: {len(train_set)}, Validation batch: {len(validation_set)}, Test batch: {len(test_set)}""")
    # trainer = Trainer(options=options)
    # trainer.train(train_set, validation_set)

    # evaluator = Evaluator(options)
    # evaluator.eval(test_set, "best_checkpoint.pt")


    training_visualization(file_name="training_log.json", metrics_lst   =["loss", "acc", "f1"])
    return None


if __name__ == '__main__':
    main()
