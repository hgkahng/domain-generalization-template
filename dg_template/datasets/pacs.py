import os
import time
import glob
import typing
import pathlib

import gdown
import tarfile

import numpy as np
import torch

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Subset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

from dg_template.datasets.base import MultipleDomainDataset


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class PACS(MultipleDomainDataset):

    _url = "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"
    _env_mapper = {
        'P': ('photo', 0),
        'A': ('art_painting', 1),
        'C': ('cartoon', 2),
        'S': ('sketch', 3),
    }

    def __init__(self,
                 root: str = 'data/domainbed/pacs/',
                 train_environments: typing.List[str] = ['P', 'A', 'C'],
                 test_environments: typing.List[str] = ['S'],
                 holdout_fraction: float = 0.2,  # size of ID validation data
                 download: bool = False
                 ) -> None:
        
        super(PACS, self).__init__()
        self.root = root
        self.train_environments = train_environments
        self.test_environments = test_environments
        self.holdout_fraction = holdout_fraction

        if download and (not os.path.exists(f'{self.root}/PACS.zip')):
            self._download()

        # find all JPG files
        input_files = np.array(
            glob.glob(os.path.join(self.root, "**/*.jpg"), recursive=True)
        )

        # find environment names (i.e., photo, art_painting, cartoon, sketch)
        env_strings = np.array([pathlib.Path(f).parent.parent.name for f in input_files])

        # create {train, val} datasets for each domain
        self._train_datasets = []
        self._id_validation_datasets = []
        for env in self.train_environments:

            # domain mask (& indices)
            env_str, _ = self._env_mapper[env]
            env_mask = (env_strings == env_str)
            env_indices = np.where(env_mask)[0]

            # FIXME: stratify with labels?
            # train, validation mask (& indices)
            np.random.shuffle(env_indices);  # TODO: set random seed
            split_idx = int(self.holdout_fraction * len(env_indices))
            val_indices = env_indices[:split_idx]
            train_indices = env_indices[split_idx:]

            self._train_datasets += []          # TODO:
            self._id_validation_datasets += []  # TODO: 

        # create test dataset
        self._test_datasets = list()
        for env in self.test_environments:

            # domain mask
            env_str, _ = self._env_mapper[env]
            env_mask = (env_strings == env_str)

            self._test_datasets += []  # TODO:

        # domains as integer values
        self.train_domains = [self._env_mapper[env][1] for env in self.train_environments]
        self.test_domains = [self._env_mapper[env][1] for env in self.test_environments]

    def _download(self) -> None:
        
        os.makedirs(self.root, exist_ok=True)
        
        # download
        _dst = os.path.join(self.root, 'PACS.zip')
        if not os.path.exists(_dst):
            gdown.download(self._url, _dst, quiet=False)
        
        # extract
        from zipfile import ZipFile
        zf = ZipFile(_dst, "r")
        zf.extractall(os.path.dirname(_dst))
        zf.close()
        
        # rename directory
        if os.path.isdir(os.path.join(self.root, 'kfold')):
            os.rename(
                src=os.path.join(self.root, "kfold"),
                dst=os.path.join(self.root, 'PACS')
            )


class SinglePACS(torch.utils.data.Dataset):

    _size: tuple = (224, 224)
    _allowed_labels = ('bird', 'car', 'chair', 'dog', 'person')  # FIXME: 
    _allowed_domains = ('P', 'A', 'C', 'S')

    def __init__(self, input_files: np.ndarray):

        super().__init__()
        self.input_files = input_files
        self.resize_fn = Resize(self._size)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.input_files)
