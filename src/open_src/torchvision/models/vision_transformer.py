"""
This module from torchvision model with customization in
    custom_vit_b_16(),
    custom_vit_b_32(),
    custom_vit_l_16(),
    custom_vit_l_32(),
    custom_vit_h_14()
    -> Modify the way paras is passed into _vision_transformer()
"""
from typing import Any, Optional
from torchvision.models.vision_transformer import (
    VisionTransformer,
    _vision_transformer,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
    ViT_H_14_Weights
)

__all__ = [
    "custom_vit_b_16", "ViT_B_16_Weights",
    "custom_vit_b_32", "ViT_B_32_Weights",
    "custom_vit_l_16", "ViT_L_16_Weights",
    "custom_vit_l_32", "ViT_L_32_Weights",
    "custom_vit_h_14", "ViT_H_14_Weights",
]


def custom_vit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

     autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    weights = ViT_B_16_Weights.verify(weights)
    return _vision_transformer(
        patch_size=kwargs.pop("patch_size", 16),
        num_layers=kwargs.pop("num_layers", 12),
        num_heads=kwargs.pop("num_heads", 12),
        hidden_dim=kwargs.pop("hidden_dim", 768),
        mlp_dim=kwargs.pop("mlp_dim", 3072),
        weights=weights,
        progress=progress,
        **kwargs,
    )


def custom_vit_b_32(*, weights: Optional[ViT_B_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    autoclass:: torchvision.models.ViT_B_32_Weights
        :members:
    """
    weights = ViT_B_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=kwargs.pop("patch_size", 32),
        num_layers=kwargs.pop("num_layers", 12),
        num_heads=kwargs.pop("num_heads", 12),
        hidden_dim=kwargs.pop("hidden_dim", 768),
        mlp_dim=kwargs.pop("mlp_dim", 3072),
        weights=weights,
        progress=progress,
        **kwargs,
    )


def custom_vit_l_16(*, weights: Optional[ViT_L_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    autoclass:: torchvision.models.ViT_L_16_Weights
        :members:
    """
    weights = ViT_L_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=kwargs.pop("patch_size", 16),
        num_layers=kwargs.pop("num_layers", 24),
        num_heads=kwargs.pop("num_heads", 16),
        hidden_dim=kwargs.pop("hidden_dim", 1024),
        mlp_dim=kwargs.pop("mlp_dim", 4096),
        weights=weights,
        progress=progress,
        **kwargs,
    )


def custom_vit_l_32(*, weights: Optional[ViT_L_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    autoclass:: torchvision.models.ViT_L_32_Weights
        :members:
    """
    weights = ViT_L_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=kwargs.pop("patch_size", 32),
        num_layers=kwargs.pop("num_layers", 24),
        num_heads=kwargs.pop("num_heads", 16),
        hidden_dim=kwargs.pop("hidden_dim", 1024),
        mlp_dim=kwargs.pop("mlp_dim", 4096),
        weights=weights,
        progress=progress,
        **kwargs,
    )


def custom_vit_h_14(*, weights: Optional[ViT_H_14_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_h_14 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_H_14_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_H_14_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    autoclass:: torchvision.models.ViT_H_14_Weights
        :members:
    """
    weights = ViT_H_14_Weights.verify(weights)

    return _vision_transformer(
        patch_size=kwargs.pop("patch_size", 14),
        num_layers=kwargs.pop("num_layers", 32),
        num_heads=kwargs.pop("num_heads", 16),
        hidden_dim=kwargs.pop("hidden_dim", 1280),
        mlp_dim=kwargs.pop("mlp_dim", 5120),
        weights=weights,
        progress=progress,
        **kwargs,
    )
