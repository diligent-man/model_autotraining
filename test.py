# This file is used for testing syntax
import math
import torch


def main() -> None:
    # Your code here
    a = torch.arange(start=0, end=1, step=round(1/9, 1))
    print(a)
    return None


if __name__ == '__main__':
    main()