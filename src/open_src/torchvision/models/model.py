from torchvision.models import (
    alexnet, AlexNet_Weights,
    googlenet, GoogLeNet_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_small, ConvNeXt_Small_Weights,
    convnext_large, ConvNeXt_Large_Weights,
    densenet121, DenseNet121_Weights,
    densenet161, DenseNet161_Weights,
    densenet169, DenseNet169_Weights,
    densenet201, DenseNet201_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    efficientnet_b5, EfficientNet_B5_Weights,
    efficientnet_b6, EfficientNet_B6_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
    inception_v3, Inception_V3_Weights,
    mnasnet0_5, MNASNet0_5_Weights,
    mnasnet0_75, MNASNet0_75_Weights,
    mnasnet1_0, MNASNet1_0_Weights,
    mnasnet1_3, MNASNet1_3_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    regnet_y_400mf, RegNet_Y_400MF_Weights,
    regnet_y_800mf, RegNet_Y_800MF_Weights,
    regnet_y_1_6gf, RegNet_Y_1_6GF_Weights,
    regnet_y_3_2gf, RegNet_Y_3_2GF_Weights,
    regnet_y_8gf, RegNet_Y_8GF_Weights,
    regnet_y_16gf, RegNet_Y_16GF_Weights,
    regnet_y_32gf, RegNet_Y_32GF_Weights,
    regnet_y_128gf, RegNet_Y_128GF_Weights,
    regnet_x_400mf, RegNet_X_400MF_Weights,
    regnet_x_800mf, RegNet_X_800MF_Weights,
    regnet_x_1_6gf, RegNet_X_1_6GF_Weights,
    regnet_x_3_2gf, RegNet_X_3_2GF_Weights,
    regnet_x_8gf, RegNet_X_8GF_Weights,
    regnet_x_16gf, RegNet_X_16GF_Weights,
    regnet_x_32gf, RegNet_X_32GF_Weights,
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights,
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
    resnext101_32x8d, ResNeXt101_32X8D_Weights,
    resnext101_64x4d, ResNeXt101_64X4D_Weights,
    wide_resnet50_2, Wide_ResNet50_2_Weights,
    wide_resnet101_2, Wide_ResNet101_2_Weights,
    shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights,
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,
    shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights,
    shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights,
    squeezenet1_0, SqueezeNet1_0_Weights,
    squeezenet1_1, SqueezeNet1_1_Weights,
    vgg11, VGG11_Weights,
    vgg11_bn, VGG11_BN_Weights,
    vgg13, VGG13_Weights,
    vgg13_bn, VGG13_BN_Weights,
    vgg16, VGG16_Weights,
    vgg16_bn, VGG16_BN_Weights,
    vgg19, VGG19_Weights,
    vgg19_bn, VGG19_BN_Weights,
    vit_b_16, ViT_B_16_Weights,
    vit_b_32, ViT_B_32_Weights,
    vit_l_16, ViT_L_16_Weights,
    vit_l_32, ViT_L_32_Weights,
    vit_h_14, ViT_H_14_Weights,
    swin_t, Swin_T_Weights,
    swin_s, Swin_S_Weights,
    swin_b, Swin_B_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    swin_v2_s, Swin_V2_S_Weights,
    swin_v2_b, Swin_V2_B_Weights,
    maxvit_t, MaxVit_T_Weights
)

available_model = {
    "alexnet": alexnet,
    "googlenet": googlenet,
    "convnext_base": convnext_base,
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_large": convnext_large,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "densenet201": densenet201,
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "efficientnet_b4": efficientnet_b4,
    "efficientnet_b5": efficientnet_b5,
    "efficientnet_b6": efficientnet_b6,
    "efficientnet_b7": efficientnet_b7,
    "efficientnet_v2_s": efficientnet_v2_s,
    "efficientnet_v2_m": efficientnet_v2_m,
    "efficientnet_v2_l": efficientnet_v2_l,
    "inception_v3": inception_v3,
    "mnasnet0_5": mnasnet0_5,
    "mnasnet0_75": mnasnet0_75,
    "mnasnet1_0": mnasnet1_0,
    "mnasnet1_3": mnasnet1_3,
    "mobilenet_v2": mobilenet_v2,
    "mobilenet_v3_small": mobilenet_v3_small,
    "mobilenet_v3_large": mobilenet_v3_large,
    "regnet_y_400mf": regnet_y_400mf,
    "regnet_y_800mf": regnet_y_800mf,
    "regnet_y_1_6gf": regnet_y_1_6gf,
    "regnet_y_3_2gf": regnet_y_3_2gf,
    "regnet_y_8gf": regnet_y_8gf,
    "regnet_y_16gf": regnet_y_16gf,
    "regnet_y_32gf": regnet_y_32gf,
    "regnet_y_128gf": regnet_y_128gf,
    "regnet_x_400mf": regnet_x_400mf,
    "regnet_x_800mf": regnet_x_800mf,
    "regnet_x_1_6gf": regnet_x_1_6gf,
    "regnet_x_3_2gf": regnet_x_3_2gf,
    "regnet_x_8gf": regnet_x_8gf,
    "regnet_x_16gf": regnet_x_16gf,
    "regnet_x_32gf": regnet_x_32gf,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x8d": resnext101_32x8d,
    "resnext101_64x4d": resnext101_64x4d,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
    "shufflenet_v2_x0_5": shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": shufflenet_v2_x1_0,
    "shufflenet_v2_x1_5": shufflenet_v2_x1_5,
    "shufflenet_v2_x2_0": shufflenet_v2_x2_0,
    "squeezenet1_0": squeezenet1_0,
    "squeezenet1_1": squeezenet1_1,
    "vgg11": vgg11,
    "vgg11_bn": vgg11_bn,
    "vgg13": vgg13,
    "vgg13_bn": vgg13_bn,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "vgg19": vgg19,
    "vgg19_bn": vgg19_bn,
    "vit_b_16": vit_b_16,
    "vit_b_32": vit_b_32,
    "vit_l_16": vit_l_16,
    "vit_l_32": vit_l_32,
    "vit_h_14": vit_h_14,
    "swin_t": swin_t,
    "swin_s": swin_s,
    "swin_b": swin_b,
    "swin_v2_t": swin_v2_t,
    "swin_v2_s": swin_v2_s,
    "swin_v2_b": swin_v2_b,
    "maxvit_t": maxvit_t
}

available_weight = {
    "alexnet": AlexNet_Weights.DEFAULT,
    "googlenet": GoogLeNet_Weights.DEFAULT,
    "convnext_base": ConvNeXt_Base_Weights.DEFAULT,
    "convnext_tiny": ConvNeXt_Tiny_Weights.DEFAULT,
    "convnext_small": ConvNeXt_Small_Weights.DEFAULT,
    "convnext_large": ConvNeXt_Large_Weights.DEFAULT,
    "densenet121": DenseNet121_Weights.DEFAULT,
    "densenet161": DenseNet161_Weights.DEFAULT,
    "densenet169": DenseNet169_Weights.DEFAULT,
    "densenet201": DenseNet201_Weights.DEFAULT,
    "efficientnet_b0": EfficientNet_B0_Weights.DEFAULT,
    "efficientnet_b1": EfficientNet_B1_Weights.DEFAULT,
    "efficientnet_b2": EfficientNet_B2_Weights.DEFAULT,
    "efficientnet_b3": EfficientNet_B3_Weights.DEFAULT,
    "efficientnet_b4": EfficientNet_B4_Weights.DEFAULT,
    "efficientnet_b5": EfficientNet_B5_Weights.DEFAULT,
    "efficientnet_b6": EfficientNet_B6_Weights.DEFAULT,
    "efficientnet_b7": EfficientNet_B7_Weights.DEFAULT,
    "efficientnet_v2_s": EfficientNet_V2_S_Weights.DEFAULT,
    "efficientnet_v2_m": EfficientNet_V2_M_Weights.DEFAULT,
    "efficientnet_v2_l": EfficientNet_V2_L_Weights.DEFAULT,
    "inception_v3": Inception_V3_Weights.DEFAULT,
    "mnasnet0_5": MNASNet0_5_Weights.DEFAULT,
    "mnasnet0_75": MNASNet0_75_Weights.DEFAULT,
    "mnasnet1_0": MNASNet1_0_Weights.DEFAULT,
    "mnasnet1_3": MNASNet1_3_Weights.DEFAULT,
    "mobilenet_v2": MobileNet_V2_Weights.DEFAULT,
    "mobilenet_v3_small": MobileNet_V3_Small_Weights.DEFAULT,
    "mobilenet_v3_large": MobileNet_V3_Large_Weights.DEFAULT,
    "regnet_y_400mf": RegNet_Y_400MF_Weights.DEFAULT,
    "regnet_y_800mf": RegNet_Y_800MF_Weights.DEFAULT,
    "regnet_y_1_6gf": RegNet_Y_1_6GF_Weights.DEFAULT,
    "regnet_y_3_2gf": RegNet_Y_3_2GF_Weights.DEFAULT,
    "regnet_y_8gf": RegNet_Y_8GF_Weights.DEFAULT,
    "regnet_y_16gf": RegNet_Y_16GF_Weights.DEFAULT,
    "regnet_y_32gf": RegNet_Y_32GF_Weights.DEFAULT,
    "regnet_y_128gf": RegNet_Y_128GF_Weights.DEFAULT,
    "regnet_x_400mf": RegNet_X_400MF_Weights.DEFAULT,
    "regnet_x_800mf": RegNet_X_800MF_Weights.DEFAULT,
    "regnet_x_1_6gf": RegNet_X_1_6GF_Weights.DEFAULT,
    "regnet_x_3_2gf": RegNet_X_3_2GF_Weights.DEFAULT,
    "regnet_x_8gf": RegNet_X_8GF_Weights.DEFAULT,
    "regnet_x_16gf": RegNet_X_16GF_Weights.DEFAULT,
    "regnet_x_32gf": RegNet_X_32GF_Weights.DEFAULT,
    "resnet18": ResNet18_Weights.DEFAULT,
    "resnet34": ResNet34_Weights.DEFAULT,
    "resnet50": ResNet50_Weights.DEFAULT,
    "resnet101": ResNet101_Weights.DEFAULT,
    "resnet152": ResNet152_Weights.DEFAULT,
    "resnext50_32x4d": ResNeXt50_32X4D_Weights.DEFAULT,
    "resnext101_32x8d": ResNeXt101_32X8D_Weights.DEFAULT,
    "resnext101_64x4d": ResNeXt101_64X4D_Weights.DEFAULT,
    "wide_resnet50_2": Wide_ResNet50_2_Weights.DEFAULT,
    "wide_resnet101_2": Wide_ResNet101_2_Weights.DEFAULT,
    "shufflenet_v2_x0_5": ShuffleNet_V2_X0_5_Weights.DEFAULT,
    "shufflenet_v2_x1_0": ShuffleNet_V2_X1_0_Weights.DEFAULT,
    "shufflenet_v2_x1_5": ShuffleNet_V2_X1_5_Weights.DEFAULT,
    "shufflenet_v2_x2_0": ShuffleNet_V2_X2_0_Weights.DEFAULT,
    "squeezenet1_0": SqueezeNet1_0_Weights.DEFAULT,
    "squeezenet1_1": SqueezeNet1_1_Weights.DEFAULT,
    "vgg11": VGG11_Weights.DEFAULT,
    "vgg11_bn": VGG11_BN_Weights.DEFAULT,
    "vgg13": VGG13_Weights.DEFAULT,
    "vgg13_bn": VGG13_BN_Weights.DEFAULT,
    "vgg16": VGG16_Weights.DEFAULT,
    "vgg16_bn": VGG16_BN_Weights.DEFAULT,
    "vgg19": VGG19_Weights.DEFAULT,
    "vgg19_bn": VGG19_BN_Weights.DEFAULT,
    "vit_b_16": ViT_B_16_Weights.DEFAULT,
    "vit_b_32": ViT_B_32_Weights.DEFAULT,
    "vit_l_16": ViT_L_16_Weights.DEFAULT,
    "vit_l_32": ViT_L_32_Weights.DEFAULT,
    "vit_h_14": ViT_H_14_Weights.DEFAULT,
    "swin_t": Swin_T_Weights.DEFAULT,
    "swin_s": Swin_S_Weights.DEFAULT,
    "swin_b": Swin_B_Weights.DEFAULT,
    "swin_v2_t": Swin_V2_T_Weights.DEFAULT,
    "swin_v2_s": Swin_V2_S_Weights.DEFAULT,
    "swin_v2_b": Swin_V2_B_Weights.DEFAULT,
    "maxvit_t": MaxVit_T_Weights.DEFAULT
}