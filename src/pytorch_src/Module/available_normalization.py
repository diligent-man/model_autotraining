from torch.nn import LocalResponseNorm, CrossMapLRN2d, LayerNorm, GroupNorm

available_normalization = {
    "LocalResponseNorm": LocalResponseNorm,
    "CrossMapLRN2d": CrossMapLRN2d,
    "LayerNorm": LayerNorm,
    "GroupNorm": GroupNorm
}
