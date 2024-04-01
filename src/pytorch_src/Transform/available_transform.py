from torchvision.transforms.v2 import (
    # Color
    ColorJitter,
    Grayscale,
    RandomAdjustSharpness,
    RandomAutocontrast,
    RandomChannelPermutation,
    RandomEqualize,
    RandomGrayscale,
    RandomInvert,
    RandomPhotometricDistort,
    RandomPosterize,
    RandomSolarize,

    # Geometry
    CenterCrop,
    ElasticTransform,
    FiveCrop,
    Pad,
    RandomAffine,
    RandomCrop,
    RandomHorizontalFlip,
    RandomIoUCrop,
    RandomPerspective,
    RandomResize,
    RandomResizedCrop,
    RandomRotation,
    RandomShortestSize,
    RandomVerticalFlip,
    RandomZoomOut,
    Resize,
    ScaleJitter,
    TenCrop,

    # Meta
    ClampBoundingBoxes,
    ConvertBoundingBoxFormat,

    # Misc
    ConvertImageDtype,
    GaussianBlur,
    Identity,
    Lambda,
    LinearTransformation,
    Normalize,
    SanitizeBoundingBoxes,
    ToDtype,

    # Temporal
    UniformTemporalSubsample,

    # Type conversion
    PILToTensor,
    ToImage,
    ToPILImage,
    ToPureTensor,
    ToTensor
)

available_transform = {
    # Color
    "ColorJitter": ColorJitter,
    "Grayscale": Grayscale,
    "RandomAdjustSharpness": RandomAdjustSharpness,
    "RandomAutocontrast": RandomAutocontrast,
    "RandomChannelPermutation": RandomChannelPermutation,
    "RandomEqualize": RandomEqualize,
    "RandomGrayscale": RandomGrayscale,
    "RandomInvert": RandomInvert,
    "RandomPhotometricDistort": RandomPhotometricDistort,
    "RandomPosterize": RandomPosterize,
    "RandomSolarize": RandomSolarize,

    # Geometry
    "CenterCrop": CenterCrop,
    "ElasticTransform": ElasticTransform,
    "FiveCrop": FiveCrop,
    "Pad": Pad,
    "RandomAffine": RandomAffine,
    "RandomCrop": RandomCrop,
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "RandomIoUCrop": RandomIoUCrop,
    "RandomPerspective": RandomPerspective,
    "RandomResize": RandomResize,
    "RandomResizedCrop": RandomResizedCrop,
    "RandomRotation": RandomRotation,
    "RandomShortestSize": RandomShortestSize,
    "RandomVerticalFlip": RandomVerticalFlip,
    "RandomZoomOut": RandomZoomOut,
    "Resize": Resize,
    "ScaleJitter": ScaleJitter,
    "TenCrop": TenCrop,

    # Meta
    "ClampBoundingBoxes": ClampBoundingBoxes,
    "ConvertBoundingBoxFormat": ConvertBoundingBoxFormat,

    # Misc
    "ConvertImageDtype": ConvertImageDtype,
    "GaussianBlur": GaussianBlur,
    "Identity": Identity,
    "Lambda": Lambda,
    "LinearTransformation": LinearTransformation,
    "Normalize": Normalize,
    "SanitizeBoundingBoxes": SanitizeBoundingBoxes,
    "ToDtype": ToDtype,

    # Temporal
    "UniformTemporalSubsample": UniformTemporalSubsample,

    # Type conversion
    "PILToTensor": PILToTensor,
    "ToImage": ToImage,
    "ToPILImage": ToPILImage,
    "ToPureTensor": ToPureTensor,
    "ToTensor": ToTensor
}