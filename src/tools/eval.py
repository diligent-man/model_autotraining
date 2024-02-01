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
                 log_path=os.path.join(os.getcwd(), "logs", "eval_log.json"),
                 checkpoint_path=os.path.join(os.getcwd(), "checkpoints"),
                 map_location="cuda" if torch.cuda.is_available() else "cpu"):
        self.__options = options
        self.__log_path = log_path
        self.__checkpoint_path = checkpoint_path
        self.__checkpoint = torch.load(f=os.path.join(self.__checkpoint_path, "best_checkpoint.pt"), map_location=map_location)


    def eval(self, test_set: DataLoader, checkpoint_name: str,
             device="cuda" if torch.cuda.is_available() else "cpu") -> None:
        model = get_model(self.__checkpoint["model_state_dict"], **self.__options.NN)
        model.eval()
        multiclass_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.__options.NN.NUM_CLASSES, device=device)  # row for actual, col for predicted

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








