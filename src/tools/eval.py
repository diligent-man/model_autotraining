import os
from typing import List

from pprint import pprint as pp
from tqdm import tqdm
from src.modelling.vgg import get_vgg_model
from src.modelling.resnet import get_resnet_model
from src.utils.logger import Logger

import torch
from torch.utils.data import DataLoader
from sklearn.metrics._classification import classification_report, precision_score, recall_score


class Evaluator:
    def __init__(self, options: dict, log_path: str, checkpoint_path: str):
        self.__options = options
        self.__log_path = log_path
        self.__checkpoint_path = checkpoint_path

        self.__model = self.__init_model()
    # Setter & Getter
    @property
    def model(self):
        return self.__model


    # Public methods
    def eval(self, test_set: DataLoader) -> None:
        print("Start evaluating model")
        self.__model.eval()
        # Init
        logger = Logger(log_path=self.__log_path)

        ground_truth_lst = []
        prediction_lst = []
        with torch.no_grad():
            for index, batch in tqdm(enumerate(test_set), total=len(test_set), desc="Evaluating"):
                imgs, ground_truths = batch[0].type(torch.FloatTensor), batch[1] #torch.reshape(input=, shape=(1, -1))
                if self.__options.MISC.CUDA:
                    imgs = imgs.to("cuda")
                    ground_truths = ground_truths.to("cuda")

                predictions = self.__model(imgs)
                # Threshoding predictions
                
                print(predictions)
                predictions = torch.argmax(input=predictions, dim=1)

                ground_truth_lst += ground_truths.detach().cpu().numpy().tolist()
                prediction_lst += predictions.detach().cpu().numpy().tolist()

        conf_report = classification_report(y_true=ground_truth_lst, y_pred=prediction_lst, digits=4, output_dict=False, zero_division="warn")
        precision = precision_score(y_true=ground_truth_lst, y_pred=prediction_lst, average="macro")
        recall = recall_score(y_true=ground_truth_lst, y_pred=prediction_lst, average="macro")

        print(conf_report)
        print(precision)
        print(recall)
        return None

    # Private methods
    def __init_model(self) -> torch.nn.Module:
        available_bases = {
            "vgg": get_vgg_model,
            "resnet": get_resnet_model
        }
        assert self.__options.MODEL.BASE in available_bases.keys(), "Your selected base is unavailable"

        checkpoint = torch.load(f=os.path.join(self.__checkpoint_path, self.__options.CHECKPOINT.NAME),
                                map_location="cuda") if self.__options.MISC.CUDA else torch.load(self.__checkpoint_path, map_location="cpu")
        model = available_bases[self.__options.MODEL.BASE](self.__options.MISC.CUDA, self.__options.MODEL.PRETRAINED,
                                                           self.__options.MODEL.NAME, checkpoint["model_state_dict"],
                                                           **self.__options.MODEL.ARGS)
        return model
