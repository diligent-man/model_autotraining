import os
from typing import List

from tqdm import tqdm
from src.modelling.vgg import get_vgg_model
from src.modelling.resnet import get_resnet_model

import torch
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassConfusionMatrix


class Evaluator:
    def __init__(self, options: dict, log_path: str, checkpoint_path: str):
        self.__options = options
        self.__log_path = log_path
        self.__checkpoint_path = checkpoint_path

        self.__model = self.__init_model(cuda=self.__options.MISC.CUDA, pretrained=self.__options.MODEL.PRETRAINED,
                                         base=self.__options.MODEL.BASE, name=self.__options.MODEL.NAME,
                                         checkpoint_path=self.__checkpoint_path, checkpoint_name=self.__options.CHECKPOINT.NAME,
                                         **self.__options.MODEL.ARGS)
        self.__metrics = self.__init_metrics(cuda=self.__options.MISC.CUDA,
                                             metrics_lst=self.__options.METRICS.NAME_LIST,
                                             args=self.__options.METRICS.ARGS)
        # multiclass_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.__options.NN.NUM_CLASSES, device="cpu")  # row for actual, col for predicted

    @staticmethod
    def __init_metrics(cuda: bool, metrics_lst: List, args: dict) -> List:
        return
    # Setter & Getter
    @property
    def model(self):
        return self.__model

    # Public methods
    def eval(self, test_set: DataLoader) -> None:
        print("Start evaluating model")
        self.__model.eval()

        with torch.no_grad():
            for index, batch in tqdm(enumerate(test_set), total=len(test_set), desc="Evaluating"):
                imgs, ground_truths = batch[0].type(torch.FloatTensor), batch[1]

                if self.__options.MISC.CUDA:
                    imgs = imgs.to("cuda")
                    ground_truths = ground_truths.to("cuda")

                predictions = self.__model(imgs)
                multiclass_confusion_matrix.update(predictions, ground_truths)

        print(multiclass_confusion_matrix.state_dict())
        return None

    # Private methods
    @staticmethod
    def __init_model(cuda: bool, pretrained: bool,
                     base: str, name: str,
                     checkpoint_path: str, checkpoint_name: str,
                     **kwargs
                     ) -> torch.nn.Module:
        available_bases = {
            "vgg": get_vgg_model,
            "resnet": get_resnet_model
        }
        assert base in available_bases.keys(), "Your selected base is unavailable"
        checkpoint = torch.load(f=os.path.join(checkpoint_path, checkpoint_name), map_location="cuda") if cuda else torch.load(checkpoint_path, map_location="cpu")

        model = available_bases[base](cuda, pretrained, name, checkpoint["model_state_dict"], **kwargs)
        return model







