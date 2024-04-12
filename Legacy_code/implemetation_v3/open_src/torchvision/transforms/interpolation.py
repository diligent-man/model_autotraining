from torchvision.transforms import InterpolationMode

available_interpolation = {
    "NEAREST": InterpolationMode.NEAREST,
    "NEAREST_EXACT": InterpolationMode.NEAREST_EXACT,
    "BILINEAR": InterpolationMode.BILINEAR,
    "BICUBIC": InterpolationMode.BICUBIC,

    # For PIL compatibility
    "BOX": InterpolationMode.BOX,
    "HAMMING": InterpolationMode.HAMMING,
    "LANCZOS": InterpolationMode.LANCZOS,
}