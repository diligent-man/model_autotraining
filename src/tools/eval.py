import os

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.modelling.vgg import get_model
from sklearn.metrics import precision_recall_curve

import torch
from torcheval.metrics import MulticlassConfusionMatrix
from torchsummary import summary


class Evaluator:
    def __init__(self, options: dict,
                 log_path=os.path.join(os.getcwd(), "logs", "eval_log.json")
                 ):
        self.__options = options
        self.__log_path = log_path


    # Public methods
    def eval(self, test_set: DataLoader, checkpoint_name: str) -> None:
        if self.__options.DEVICE.CUDA:
            checkpoint = torch.load(f=os.path.join(os.getcwd(), "checkpoints", checkpoint_name), map_location="cuda")
            model = get_model(cuda=True, model_state_dict=checkpoint["model_state_dict"], **self.__options.NN)
            multiclass_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.__options.NN.NUM_CLASSES, device="cuda")  # row for actual, col for predicted
        else:
            checkpoint = torch.load(f=os.path.join(os.getcwd(), "checkpoints", checkpoint_name), map_location="cpu")
            model = get_model(cuda=True, model_state_dict=checkpoint["model_state_dict"], **self.__options.NN)
            multiclass_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.__options.NN.NUM_CLASSES, device="cpu")  # row for actual, col for predicted
        model.eval()

        with torch.no_grad():
            for index, batch in tqdm(enumerate(test_set), total=len(test_set), desc="Evaluating"):
                imgs, ground_truths = batch[0].type(torch.FloatTensor), batch[1]

                if torch.cuda.is_available():
                    imgs = imgs.to("cuda")
                    ground_truths = ground_truths.to("cuda")

                predictions = model(imgs)
                multiclass_confusion_matrix.update(predictions, ground_truths)

        print(multiclass_confusion_matrix.state_dict())
        return None








