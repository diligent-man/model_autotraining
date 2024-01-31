import os

from box import Box
from time import time
from Model_pipeline_v3.src.modelling.vgg import get_model
from Model_pipeline_v3.src.utils.logger import Logger


import torch

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary


class Trainer:
    __options: Box

    def __init__(self, options, train_set: DataLoader, validation_set: DataLoader):
        self.__options = options
        self.__checkpoint = None

        if self.__options.CHECKPOINT.LOAD:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            self.__checkpoint = torch.load(f=os.path.join(self.__options.CHECKPOINT.NAME,
                                                          self.__options.CHECKPOINT.RESUME_NAME),
                                           map_location=map_location
                                           )
        self.__model = self.__init_model()
        self.__optimizer = self.__init_optimizer()

    # Public methods
    def train(self, train_set: DataLoader, test_set: DataLoader) -> None:
        print("Start training model ...")
        self.__model.train()

        logger = Logger(log_path=self.__path_options["log_path"])

        # epoch = self.__epoch_options["start_epoch"]
        #
        # if self.__checkpoint_options["load_checkpoint"]:
        #     epoch = self.__checkpoint["epoch"] + 1
        #
        # # Start training
        # for epoch in range(epoch, epoch + self.__epoch_options["epochs"]):
        #     start_time = time()
        #     total_sample = 0  # for computing avg of metrics
        #
        #     for index, batch in tqdm(enumerate(train_set), total=len(train_set), colour="cyan", desc="Training"):
        #         imgs, labels = batch[0].type(torch.FloatTensor) / 255, batch[1]
        #
        #         # Pass to predefined device
        #         if self.__device_options["cuda"]:
        #             imgs = imgs.to(self.__device_options["device"])
        #             labels = labels.to(self.__device_options["device"])
        #
        #         # clear gradients of all vars (weights, biases) before performing backprop
        #         self.__optimizer.zero_grad()
        #
        #         # Inference
        #         predictions = self.__model(imgs)
        #
        #         # compute loss & backprop
        #         loss = CrossEntropyLoss()(predictions, labels)
        #         loss.backward()
        #
        #         # Adjust optimizer paras
        #         self.__optimizer.step()
        #
        #         # Accumulate metrics after each iteration
        #         self.__computeAccumulativeMetrics(predictions=predictions, labels=labels, flag="train")
        #
        #         # for key in self.__metric_options["metrics"]:
        #         #     print(key, self.__metric_options["train_metrics"][key])
        #
        #     # Take mean of metrics
        #     for key in self.__metric_options["metrics"]:
        #         self.__metric_options["train_metrics"][key] = 100 * self.__metric_options["train_metrics"][key] / total_sample
        #
        #     # Evaluate model
        #     self.__eval(test_set)
        #
        #     # Logging
        #     logger.write_to_file(**{"epoch": epoch,
        #                             "time per epoch": time() - start_time,
        #                             "loss": loss.item(),
        #
        #                             "train_accuracy": self.__metric_options["train_metrics"]["accuracy"],
        #                             "train_recall": self.__metric_options["train_metrics"]["recall"],
        #                             "train_precision": self.__metric_options["train_metrics"]["precision"],
        #
        #                             "eval_accuracy": self.__metric_options["eval_metrics"]["accuracy"],
        #                             "eval_recall": self.__metric_options["eval_metrics"]["recall"],
        #                             "eval_precision": self.__metric_options["eval_metrics"]["precision"]
        #                             }
        #                          )
        #     # Reset metrics
        #     for key1 in ("train", "eval"):
        #         for key2 in self.__metric_options[f"{key1}_metrics"].keys():
        #             self.__metric_options[f"{key1}_metrics"][key2] = .0
        #
        #     # Save model
        #     if self.__checkpoint_options["save_checkpoint"]:
        #         torch.save(obj={"epoch": epoch,
        #                         "model_state_dict": self.__model.state_dict(),
        #                         "adam_state_dict": self.__optimizer.state_dict()
        #                         }, f=os.path.join(self.__path_options["model_path"], f"epoch_{epoch}.pt")
        #                    )

            # for key in self.__metric_options["metrics"]:
            #     print(key, self.__metric_options["train_metrics"][key])
            # print(total_sample)
            # print()
            # print()
            # print()
        return None

    def get_model_summary(self):
        input_size = (
                      self.__options.DATA.INPUT_SHAPE[2],
                      self.__options.DATA.INPUT_SHAPE[0],
                      self.__options.DATA.INPUT_SHAPE[1]
                    )
        return summary(self.__model, input_size=input_size)

    # Private methods
    def __init_model(self):
        if self.__checkpoint is not None:
            model = get_model(dropout=self.__options.NN.DROPOUT,
                              model_state_dict=self.__checkpoint["model_state_dict"]
                              )
        else:
            model = get_model(dropout=self.__options.NN.DROPOUT)
        return model

    def __init_optimizer(self):
        optimizer = Adam(params=self.__model.parameters(),
                         lr=self.__options.OPTIMIZER.LR,
                         amsgrad=True
                    )

        if self.__checkpoint is not None:
            optimizer.load_state_dict(self.__checkpoint["adam_state_dict"])

        return optimizer
