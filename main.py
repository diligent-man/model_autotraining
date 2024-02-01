import os

from box import Box
from pprint import pprint as pp
from src.tools.train import Trainer
from src.utils.utils import json_decoder, get_dataset


def main() -> None:
    options = Box(next(iter(json_decoder(open("config.json").read()))))

    train_set, validation_set, test_set = get_dataset(root=os.path.join(os.getcwd(), options.DATA.PROJECT_NAME),
                                                      img_size=options.DATA.INPUT_SHAPE[0],
                                                      train_size=options.DATA.TRAIN_SIZE,
                                                      batch_size=options.DATA.BATCH_SIZE,
                                                      seed=options.MISC.SEED,
                                                      num_workers=options.DATA.NUM_WORKERS
                                                      )
    trainer = Trainer(options=options)
    trainer.train(train_set, validation_set)
    return None


if __name__ == '__main__':
    main()
