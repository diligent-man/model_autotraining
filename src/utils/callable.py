# TODO: Implement not finished
import torch


__all__ = ["available_callable"]


def to_device(x, device: str = "cpu"):
    x =x.to("cuda")
    return x


available_callable = {
    "to_device": to_device
}
