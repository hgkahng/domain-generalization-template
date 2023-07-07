
import os
import glob
import typing
import pathlib
import shutil

import numpy as np
import torch

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

from sklearn.model_selection import train_test_split

from dg_template.datasets.base import MultipleDomainCollection
from dg_template.datasets.base import MultipleDomainData
from dg_template.datasets.base import SupervisedDataModule
from dg_template.datasets.base import SemiSupervisedDataModule


class OfficeHome(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index: int) -> dict:
        pass

    def __len__(self) -> int:
        pass


class OfficeHomeDataModule(SupervisedDataModule):
    def __init__(self):
        pass

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        pass
