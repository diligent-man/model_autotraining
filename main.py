import os
import commentjson

from box import Box
from src.tools.train import Trainer
from src.tools.eval import Evaluator
from src.tools.visualization import training_visualization
from src.utils.utils import get_train_set, get_test_set, get_model_summary


def train(option_path: str) -> None:
    # Load dataset
    options = Box(commentjson.loads(open(file=option_path, mode="r").read()))
    log_path = os.path.join(os.getcwd(), "logs", f"{options.SOLVER.MODEL.NAME}_training_log.json")
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints", options.SOLVER.MODEL.NAME)

    if not os.path.isdir(checkpoint_path):
        os.mkdir(path=checkpoint_path, mode=0x777)
        print(f"directory checkpoint for {options.SOLVER.MODEL.NAME} is created.")

    train_set, validation_set = get_train_set(root=os.path.join(os.getcwd(), options.DATA.DATASET_NAME),
                                              input_size=options.DATA.INPUT_SHAPE[0],
                                              train_size=options.DATA.TRAIN_SIZE,
                                              batch_size=options.DATA.BATCH_SIZE,
                                              seed=options.MISC.SEED, cuda=options.MISC.CUDA,
                                              num_workers=options.DATA.NUM_WORKERS)
    print(f"""Train batch: {len(train_set)}, Validation batch: {len(validation_set)}""")

    trainer = Trainer(options=options, log_path=log_path, checkpoint_path=checkpoint_path)
    trainer.train(train_set, validation_set)
    return None


def evaluate(option_path: str) -> None:
    options = Box(commentjson.loads(open(file=option_path, mode="r").read()))
    log_path = os.path.join(os.getcwd(), "logs", f"{options.SOLVER.MODEL.NAME}_eval_log.json")
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints", options.SOLVER.MODEL.NAME)

    test_set = get_test_set(root=os.path.join(os.getcwd(), options.DATA.DATASET_NAME),
                            input_size=options.DATA.INPUT_SHAPE[0],
                            batch_size=options.DATA.BATCH_SIZE,
                            seed=options.MISC.SEED, cuda=options.MISC.CUDA,
                            num_workers=options.DATA.NUM_WORKERS)
    print(f"""Test batch: {len(test_set)}""")

    evaluator = Evaluator(options=options, log_path=log_path, checkpoint_path=checkpoint_path)
    evaluator.eval(test_set)
    return None


def main() -> None:
    train(option_path=os.path.join(os.getcwd(), "configs", "vgg_train_config.json"))
    # evaluate(option_path=os.path.join(os.getcwd(), "configs", "eval_config.json"))
    
    # training_visualization(file_name="training_log.json", metrics_lst=["loss", "acc", "f1"], x_interval=2)

    # To-do list
    # Model evaluator
    # Visualization
    return None


if __name__ == '__main__':
    main()
