from torch.nn.modules import (
    L1Loss,
    NLLLoss,
    KLDivLoss,
    MSELoss,
    BCELoss, BCEWithLogitsLoss, NLLLoss2d,
    CosineEmbeddingLoss,
    CTCLoss,
    HingeEmbeddingLoss,
    MarginRankingLoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    SmoothL1Loss,
    HuberLoss,
    SoftMarginLoss,
    CrossEntropyLoss,
    TripletMarginLoss,
    TripletMarginWithDistanceLoss,
    PoissonNLLLoss,
    GaussianNLLLoss
)

available_loss = {
    "L1Loss": L1Loss,
    "NLLLoss": NLLLoss,
    "KLDivLoss": KLDivLoss,
    "MSELoss": MSELoss,
    "BCELoss": BCELoss,
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "NLLLoss2d": NLLLoss2d,
    "CosineEmbeddingLoss": CosineEmbeddingLoss,
    "CTCLoss": CTCLoss,
    "HingeEmbeddingLoss": HingeEmbeddingLoss,
    "MarginRankingLoss": MarginRankingLoss,
    "MultiLabelMarginLoss": MultiLabelMarginLoss,
    "MultiLabelSoftMarginLoss": MultiLabelSoftMarginLoss,
    "MultiMarginLoss": MultiMarginLoss,
    "SmoothL1Loss": SmoothL1Loss,
    "HuberLoss": HuberLoss,
    "SoftMarginLoss": SoftMarginLoss,
    "CrossEntropyLoss": CrossEntropyLoss,
    "TripletMarginLoss": TripletMarginLoss,
    "TripletMarginWithDistanceLoss": TripletMarginWithDistanceLoss,
    "PoissonNLLLoss": PoissonNLLLoss,
    "GaussianNLLLoss": GaussianNLLLoss
}