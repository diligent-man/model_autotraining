import os

from utils.utils import get_config


import torch

from torchsummary import summary
from torchvision.models import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,
from torchvision.models import alexnet

available_models = {
    "vgg":
        {
            "vgg11": vgg11,
            "vgg13": vgg13,
            "vgg16": vgg16,
            "vgg19": vgg19,
            "vgg11_bn": vgg11_bn,
            "vgg13_bn": vgg13_bn,
            "vgg16_bn": vgg16_bn,
            "vgg19_bn": vgg19_bn
        },
    "alexnet": alexnet
    # can add more model arch
}

available_optimizers = {
    "adam": torch.optim.Adam,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad
    # Can add more other optimizers
}


class Trainer:
    __default_config_path: str
    __custom_config_path: str

    def __init__(self,
                 custom_config_path: str,
                 default_config_path: str = r"D:\Local\Source\python\semester_6\Model_pipeline_v2\src\config\default_config.json"
                 ):
        self.__config = get_config(default_config_path, custom_config_path)
        
        self.__CHECKPOINT = None
        if self.__config.config["CHECKPOINT"]["LOAD"]:
            checkpoint_path = os.path.join(self.__config["CHECKPOINT"]["PATH"],
                                           self.__config["EXPERIMENTS"]["NAME"],
                                           self.__config["CHECKPOINT"]["NAME"]
                                           )
            self.__CHECKPOINT = torch.load(f=checkpoint_path, map_location=self.__config["DEVICE"]["DEVICE_NAME"])


        self.__model = self.__init_model()
            # self.__optimizer = OptimizerManager(configGenerator).init_optimizer(model_parameters=self.__model.parameters())
        #     self.__

    # Public methods

    # def get_model_info(self):
    #     summary(input_size=(3, self.__config["DATASET"]["IMAGE_SIZE"][0], self.__config["DATASET"]["IMAGE_SIZE"][1]), model=self.__model)
    #
    # def get_optimizer_info(self):
    #     print(self.__optimizer)


    def __init_model(self):
        assert self.__config["BASE"]
        assert self.__config["NN"]["DROPOUT"] < 1, "Drop out probability should be less than 1"

        # Load derivative of this architect

        for k, v in derivatives.items():
            if k == configGenerator.config["MODEL"]["DERIVATIVE"]:
                model = derivatives[k](pretrained=configGenerator.config["MODEL"]["LOAD_PRETRAINED"],
                                       **{"dropout": configGenerator.config["NN"]["DROPOUT"]}
                                       )
                break
        assert model is not None, "Your selected derivative is unavailable !!!"

        if configGenerator.config["MODEL"]["LOAD_PRETRAINED"]:
            return model

        # Load pretrained weights/ state_dict in local
        if model_state_dict:
            print("Loading pretrained model...")
            model.load_state_dict(model_state_dict)
            print("Finished.")
        else:
            print("Initializing parameters...")
            for para in model.parameters():
                if para.dim() > 1:
                    nn.init.xavier_uniform_(para)
            print("Finished.")

        if configGenerator.config["DEVICE"]["CUDA"]:
            model = model.to(configGenerator.config["DEVICE"]["DEVICE_NAME"])
        return model

