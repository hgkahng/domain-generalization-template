
import typing

import torch
import torch.nn as nn
import torchvision.transforms as T


class CivilCommentsTransform(nn.Module):
    """
    Note: transformations are applied inside the __getitem__ function
        of the `CivilComments` dataset.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
