import torch
import torcheval

from torcheval.metrics import MulticlassPrecision, MulticlassRecall


class Evaluator:
    __precision: torcheval.metrics.Metric
    __recall: torcheval.metrics.Metric
    __f1_score: torcheval.metrics.Metric

    # validation &* test
    def __init__(self, threshold=.5):
        self.__precision = MulticlassPrecision()
        self.__recall = MulticlassPrecision()
        self.__f1_score = MulticlassPrecision()

        if torch.cuda.is_available():
            self.__precision.to(device="cuda")
            self.__recall.to(device="cuda")
            self.__f1_score.to(device="cuda")

    def eval(self):
        self.__precision.update()



