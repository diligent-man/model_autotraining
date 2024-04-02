from torch.optim.lr_scheduler import (
    LambdaLR,
    MultiplicativeLR,
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    PolynomialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ChainedScheduler,
    SequentialLR,
    ReduceLROnPlateau,
    OneCycleLR
)

available_lr_scheduler = {
    "LambdaLR": LambdaLR,
    "MultiplicativeLR": MultiplicativeLR,
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "ConstantLR": ConstantLR,
    "LinearLR": LinearLR,
    "ExponentialLR": ExponentialLR,
    "PolynomialLR": PolynomialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
    "ChainedScheduler": ChainedScheduler,
    "SequentialLR": SequentialLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "OneCycleLR": OneCycleLR
}
