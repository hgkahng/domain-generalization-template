import os
import time
import glob
import typing
import pathlib
import shutil

import gdown
import tarfile

import numpy as np
import torch

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

from dg_template.datasets.base import MultipleDomainCollection
from dg_template.datasets.base import MultipleDomainData


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class SingleVLCS(torch.utils.data.Dataset):

    _allowed_labels = ('bird', 'car', 'chair', 'dog', 'person')
    _allowed_domains = ('VOC2007', 'LabelMe', 'Caltech101', 'SUN09')
    _size = (224, 224)
    
    def __init__(self, input_files: typing.Union[np.ndarray, typing.List[str]]) -> None:
        
        super(SingleVLCS, self).__init__()        
        self.input_files = input_files
        self.resize_fn = Resize(self._size)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        
        filename: str = self.input_files[index]
        try:
            img = read_image(filename, mode=ImageReadMode.RGB)
        except RuntimeError as _:
            img = pil_loader(filename)
            img = pil_to_tensor(img)
        
        return dict(
            x=self.resize_fn(img),
            y=self._allowed_labels.index(pathlib.Path(filename).parent.name),               # int
            domain=self._allowed_domains.index(pathlib.Path(filename).parent.parent.name),  # int
            eval_group=0,
        )

    def __len__(self) -> int:
        return len(self.input_files)


class VLCSDataModule(MultipleDomainData):

    _url: str = "https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8"
    _dst: str = "VLCS.tar.gz"
    _env_mapper = {
        'V': ('VOC2007', 0),
        'L': ('LabelMe', 1),
        'C': ('Caltech101', 2),
        'S': ('SUN09', 3)
    }

    def __init__(self,
                 root: str,
                 train_domains: typing.List[str] = ['V', 'L', 'C'],
                 test_domains: typing.List[str] = ['S'],
                 holdout_fraction: float = 0.2,
                 download: bool = False):
        
        super().__init__()
        
        self.root = root
        self.download = download

        self.train_domains = train_domains
        self.test_domains = test_domains
        
        self.holdout_fraction = holdout_fraction

        self._train_datasets = []
        self._id_validation_datasets = []
        self._test_datasets = []

        self.input_files = glob.glob(os.path.join(self.root, "**/*.jpg"), recursive=True)
        self.input_files = np.array(self.input_files)

    def prepare_data(self):
        """Download & extract data."""
        if self.download:
            self._download_and_extract()

    def setup(self, stage: str):
        
        if stage == 'fit':
            pass

        if stage == 'test':
            pass

        if stage == 'predict':
            pass

    def train_dataloader(self, **kwargs):
        concat = ConcatDataset(self._train_datasets)
        return DataLoader(concat,
                          batch_size=self.batch_size,    # FIXME: 
                          sampler=None,                  # FIXME: stratified batch
                          num_workers=self.num_workers,  # FIXME: 
                          )

    def val_dataloader(self, **kwargs):
        concat = ConcatDataset(self._id_validation_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          )

    def test_dataloader(self, **kwargs):
        concat = ConcatDataset(self._test_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          )

    def _download_and_extract(self) -> None:
        
        # download
        os.makedirs(self.root, exist_ok=True)
        _dst = os.path.join(self.root, self._dst)
        if not os.path.exists(_dst):
            gdown.download(self._url, _dst, quiet=False)
        
        # extract
        tar = tarfile.open(_dst, "r:gz")
        tar.extractall(self.root)
        tar.close()

        # change folder hierarchy
        inner_dir = os.path.join(self.root, 'VLCS')
        for subdir in os.listdir(inner_dir):
            subdir = os.path.join(inner_dir, subdir)
            shutil.move(subdir, self.root)

        os.removedirs(inner_dir)

class VLCS(MultipleDomainCollection):
    
    _url: str = "https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8"
    _env_mapper = {
        'V': ('VOC2007', 0),
        'L': ('LabelMe', 1),
        'C': ('Caltech101', 2),
        'S': ('SUN09', 3)
    }
    
    def __init__(self,
                 root: str = 'data/domainbed/vlcs/',
                 train_environments: typing.List[str] = ['V', 'L', 'C'],
                 test_environments: typing.List[str] = ['S'],
                 holdout_fraction: float = 0.2,  # size of ID validation data
                 download: bool = False
                 ) -> None:
        
        super(VLCS, self).__init__()
        self.root = root
        self.train_environments = train_environments
        self.test_environments = test_environments
        self.holdout_fraction = holdout_fraction
        
        if download and (not os.path.exists(f'{self.root}/VLCS.tar.gz')):
            self._download()
        
        # find all JPG files
        input_files = np.array(glob.glob(os.path.join(self.root, "**/*.jpg"), recursive=True))

        # find environment names
        env_strings = np.array([pathlib.Path(f).parent.parent.name for f in input_files])
        
        # create {train, val} datasets for each domain
        self._train_datasets = list()
        self._id_validation_datasets = list()
        for env in self.train_environments:
            
            # domain mask (indices)
            env_str, _ = self._env_mapper[env]
            env_mask = (env_strings == env_str)
            env_indices = np.where(env_mask)[0]
            
            # FIXME: stratify with labels?
            # train, validation mask (indices)
            np.random.shuffle(env_indices);  # TODO: set random seed
            split_idx = int(self.holdout_fraction * len(env_indices))
            val_indices = env_indices[:split_idx]
            train_indices = env_indices[split_idx:]

            self._train_datasets += [SingleVLCS(input_files[train_indices])]
            self._id_validation_datasets += [SingleVLCS(input_files[val_indices])]

        # create test dataset
        self._test_datasets = list()
        for env in self.test_environments:

            # domain mask
            env_str, _ = self._env_mapper[env]
            env_mask = (env_strings == env_str)

            self._test_datasets += [SingleVLCS(input_files[env_mask])]

        # domains as integer values
        self.train_domains = [self._env_mapper[env][1] for env in self.train_environments]
        self.test_domains = [self._env_mapper[env][1] for env in self.test_environments]

    def _download(self) -> None:
        
        os.makedirs(self.root, exist_ok=True)
        _dst = os.path.join(self.root, 'VLCS.tar.gz')
        if not os.path.exists(_dst):
            gdown.download(self._url, _dst, quiet=False)
        
        tar = tarfile.open(_dst, "r:gz")
        tar.extractall(os.path.dirname(_dst))
        tar.close()