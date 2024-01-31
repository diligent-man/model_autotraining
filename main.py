import os

from box import Box
from pprint import pprint as pp
from src.tools.train import Trainer
from src.utils.utils import json_decoder, get_dataset


def main() -> None:
    options = Box(next(iter(json_decoder(open(os.path.join(os.getcwd(), "config.json")).read()))))
    train_set, validation_set, test_set = get_dataset(root=options.DATA.NAME,
                                                      img_size=options.DATA.INPUT_SHAPE[0],
                                                      train_size=options.DATA.TRAIN_SIZE,
                                                      batch_size=options.DATA.BATCH_SIZE,
                                                      seed=options.MISC.SEED,
                                                      num_workers=options.DATA.NUM_WORKERS
                                                      )

    trainer = Trainer(options=options, train_set=train_set, validation_set=validation_set)
    trainer.train()
    # trainer.train(train_set=train_set, test_set=test_set)
    """
    Task lists:
        + Logging with json format (Infinitt & NaN can not read when decoding in KNIME)
        + Add tensor board while training
        + Revise computation of training metrics
        + Take latest model when load_checkpoint == True
        + Save 4 best + 1 latest model
    """
    return None


if __name__ == '__main__':
    main()
