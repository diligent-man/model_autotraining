import os, copy
from pprint import pp

from src.tools.Trainer import Trainer
from src.utils.ConfigManager import ConfigManager



def train(config_manager: ConfigManager) -> None:
    # For testing
    import torch
    import gc
    bases = names = ("alexnet", "googlenet", "convnext_base", "convnext_tiny", "convnext_small", "convnext_large", "densenet121", "densenet161", "densenet169", "densenet201", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7", "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l", "inception_v3", "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large", "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf", "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf", "regnet_y_128gf", "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_3_2gf", "regnet_x_8gf", "regnet_x_16gf", "regnet_x_32gf", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d", "wide_resnet50_2", "wide_resnet101_2", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "squeezenet1_0", "squeezenet1_1", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14", "swin_t", "swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b", "maxvit_t")
    shapes = [299]
    cached_transform = copy.deepcopy(config_manager.DATA_TRANSFORM)
    pass_model = []

    for base, name in zip(bases, names):
        config_manager.MODEL_BASE = base
        config_manager.MODEL_NAME = name

        print(name)
        for shape in shapes:
            cached_transform["Resize"]["size"] = [shape, shape]

            config_manager.set("DATA_TRANSFORM", copy.deepcopy(cached_transform))
            trainer = Trainer(config_manager=config_manager)

            # print(trainer.model)
            trainer.get_model_summary()
            trainer.train(compute_metric_in_train=True)

            torch.cuda.empty_cache()
            gc.collect()
            break
        break

        pass_model.append(name)
        print("##########################################################################################3")
        print()
        print()

    # print(pass_model)
    #
    #
    # trainer = Trainer(config_manager=config_manager)
    # trainer.get_model_summary()
    # trainer.train(compute_metric_in_train=True)
    return None

#
# def test(option_path: str) -> None:
#     for dataset in (["celeb_A", "collected_v3", "collected_v4"]):
#         options = Box(commentjson.loads(open(file=option_path, mode="r").read()))
#         checkpoint_path = os.path.join(os.getcwd(), "checkpoints", options.MODEL.NAME, options.CHECKPOINT.NAME)
#
#         options.DATA.DATASET_NAME = dataset
#         log_path = os.path.join(os.getcwd(), "logs", options.MODEL.NAME, f"testing_log_{dataset}.json")
#
#         test_set = get_dataset(root=os.path.join(os.getcwd(), options.DATA.DATASET_NAME, "test"),
#                                transform=options.DATA.TRANSFORM,
#                                )
#
#         test_loader: DataLoader = get_test_loader(dataset=test_set,
#                                                   batch_size=options.DATA.BATCH_SIZE,
#                                                   cuda=options.MISC.CUDA,
#                                                   num_workers=options.DATA.NUM_WORKERS
#                                                   )
#         print(f"""Test batch: {len(test_loader)}""")
#
#         evaluate(options=options, checkpoint_path=checkpoint_path, log_path=log_path, test_loader=test_loader)
#     return None


def main() -> None:
    # generate_celeb_A_dataset()
    config_manager = ConfigManager(path=os.path.join(os.getcwd(), "configs", "vgg.json"))
    train(config_manager)
    # test(option_path=os.path  .join(os.getcwd(), "configs", "test_config.json"))
    return None


if __name__ == '__main__':
    main()


