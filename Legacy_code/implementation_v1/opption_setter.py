import os
import argparse
from argparse import Namespace
from typing import Any, Dict, Tuple

from torcheval.metrics.functional import binary_accuracy, binary_precision, binary_recall
# torcheval.metrics.functional just provide a function that computes metric for a current iteration not entire batch
# Use torcheval.metrics for batch computation

class OptionSetter:
    __parser: argparse.ArgumentParser
    def __init__(self):
        self.__parser = argparse.ArgumentParser()
        self.__miscSetup()
        self.__deviceSetup()
        self.__pathSetup()
        self.__datasetSetup()
        self.__epochSetup()
        self.__checkpointSetup()
        self.__optimizerSetup()
        self.__nnSetup()
        self.__metricSetup()

        self.__options: Namespace = self.__parser.parse_args()

    # Remover
    def removeOption(self, option: str):
        del self.__options[option]

    # Setter
    def setNewOption(self, option: str, value: Any):
        self.__options[option] = value

    # Getter
    def getMiscSetup(self):
        return {
            "SEED": self.__options.SEED
        }

    def getDeviceSetup(self):
        return {
            "cuda": self.__options.cuda,
            "device": self.__options.device
        }

    def getPathSetup(self):
        return {
            "dataset_path": self.__options.dataset_path,
            "dataset_save_path": self.__options.dataset_save_path,
            "annotation_path": self.__options.annotation_path,
            "log_path": self.__options.log_path,
            "model_path": self.__options.model_path
        }

    def getDatasetSetup(self):
        return {
            "img_size": self.__options.img_size,
            "train_size": self.__options.train_size,
            "batch_size": self.__options.batch_size  # Used for both train & test set
        }

    def getEpochSetup(self):
        return {
            "start_epoch": self.__options.start_epoch,
            "epochs": self.__options.epochs,
        }

    def getCheckpointSetup(self):
        return {
            "load_checkpoint": self.__options.load_checkpoint,
            "save_checkpoint": self.__options.save_checkpoint,
            "checkpoint_path": self.__options.checkpoint_path,
            "max_checkpoints": self.__options.max_checkpoints
        }

    def getOptimizerSetup(self):
        return {
            "lr": self.__options.lr,
            "betas": self.__options.betas,
            "eps": self.__options.eps,
            "weight_decay": self.__options.weight_decay,
            "amsgrad": self.__options.amsgrad
        }

    def getNnSetup(self):
        return {
            "dropout": self.__options.dropout
        }

    def getMetricSetup(self):
        return {
            "threshold": self.__options.threshold,
            "metrics": self.__options.metrics,
            "metric_funcs": self.__options.metric_funcs,
            "train_metrics": self.__options.train_metrics,
            "eval_metrics": self.__options.eval_metrics
        }

    def getAll(self):
        return self.__options

    # Private methods
    def __miscSetup(self):
        self.__parser.add_argument("--SEED", type=int, default=123)

    def __deviceSetup(self):
        # Check single gpu/ cpu
        self.__parser.add_argument("--cuda", type=bool, default=True)
        self.__parser.add_argument('--device', type=str, default="cuda")

    def __pathSetup(self):
        self.__parser.add_argument("--dataset_path", type=str, default=r"D:\Local\Source\python\semester_6\OJT\Model_pipeline_v1\Labelled_face_attribute_dataset_v2",)
        self.__parser.add_argument("--dataset_save_path", type=str, default=os.path.join(os.getcwd(), "Compiled_dataset"))
        self.__parser.add_argument("--annotation_path", type=str, default=os.path.join(os.getcwd(), "annotations.csv"))
        self.__parser.add_argument("--log_path", type=str, default=os.path.join(os.getcwd(), "log.json"))
        self.__parser.add_argument("--model_path", type=str, default=os.path.join(os.getcwd(), "Model"))

    def __datasetSetup(self):
        self.__parser.add_argument("--img_size", type=int, default=224)
        self.__parser.add_argument("--train_size", type=int, default=.8)
        self.__parser.add_argument("--batch_size", type=int, default=64)  # 15gb GPU RAM-- 100

    def __epochSetup(self):
        self.__parser.add_argument('--start-epoch', default=1, type=int)
        self.__parser.add_argument('--epochs', default=1, type=int)


    def __checkpointSetup(self):
        self.__parser.add_argument("--load_checkpoint", type=bool, default=False)
        self.__parser.add_argument("--save_checkpoint", type=bool, default=True)
        self.__parser.add_argument("--checkpoint_path", type=str, default=os.path.join(os.getcwd(), "Model", "epoch_1.pt"))
        self.__parser.add_argument("--max_checkpoints", type=int, default=5)

    def __optimizerSetup(self):
        # For Adam
        self.__parser.add_argument("--lr", default=.05, type=float)
        self.__parser.add_argument("--betas", default=(.9, .999), type=float)
        self.__parser.add_argument("--eps", default=1e-8, type=float)
        self.__parser.add_argument("--weight_decay", default=.001, type=float)
        self.__parser.add_argument("--amsgrad", default=True, type=bool)

    def __nnSetup(self):
        self.__parser.add_argument("--dropout", default=.5, type=float)

    def __metricSetup(self):
        self.__parser.add_argument("--threshold", default=0.5, type=float)
        self.__parser.add_argument("--metrics", default=("accuracy", "recall", "precision"), type=Tuple[str])
        self.__parser.add_argument("--metric_funcs", default={"accuracy": binary_accuracy, "precision": binary_precision, "recall": binary_recall}, type=Dict)
        self.__parser.add_argument("--train_metrics", default={"accuracy": 0, "recall": 0, "precision": 0}, type=Dict)
        self.__parser.add_argument("--eval_metrics", default={"accuracy": 0, "recall": 0, "precision": 0}, type=Dict)
