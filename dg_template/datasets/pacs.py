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

from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

from sklearn.model_selection import train_test_split

from dg_template.datasets.base import MultipleDomainCollection
from dg_template.datasets.base import SupervisedDataModule
from dg_template.datasets.base import SemiSupervisedDataModule


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class PACS(torch.utils.data.Dataset):

    _allowed_labels = ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person')
    _allowed_domains = ('photo', 'art_painting', 'cartoon', 'sketch')
    _size = (224, 224)

    _url = "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"
    _env_mapper = {
        'P': ('photo', 0),
        'A': ('art_painting', 1),
        'C': ('cartoon', 2),
        'S': ('sketch', 3),
    }

    def __init__(self,
                 root: str,
                 environments: typing.Iterable[str] = ['P', 'A', 'C', 'S'],
                 download: bool = False,
                 ) -> None:
        super().__init__()

        self.resize_fn = Resize(size=self._size)

        self.root = root
        self.environments = environments
        
        if download:
            self.download(root)

        # collect input files (X)
        self.input_files = list()
        for env in self.environments:
            pattern = os.path.join(self.root, self._env_mapper[env][0], "**/*.[jpJP][npNP]*[gG$]")
            files = glob.glob(pattern, recursive=True)
            self.input_files += files
        self.input_files = np.array(self.input_files)  # array of strings

        # collect targets (Y)
        self.labels = [pathlib.Path(filename).parent.name for filename in self.input_files]  # list of strings
        self.labels = np.array([self._allowed_labels.index(s) for s in self.labels])  # array of integers

        # collect domains (S)
        self.domains = [pathlib.Path(filename).parent.parent.name for filename in self.input_files]  # list of strings
        self.domains = np.array([self._allowed_domains.index(s) for s in self.domains])  # array of integers

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        
        filename: str = self.input_files[index]
        try:
            img = read_image(filename, mode=ImageReadMode.RGB)
        except RuntimeError as _:
            img = pil_loader(filename)
            img = pil_to_tensor(img)

        return dict(
            x=self.resize_fn(img),
            y=self.labels[index],
            domain=self.domains[index],
            eval_group=0,
        )

    def __len__(self) -> int:
        return len(self.input_files)

    @staticmethod
    def download(root: str) -> None:
        
        url = "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"
        dst = "PACS.zip"

        os.makedirs(root, exist_ok=True)
        
        # download
        _dst = os.path.join(root, dst)
        if not os.path.exists(_dst):
            gdown.download(url, _dst, quiet=False)
        
        # extract
        from zipfile import ZipFile
        zf = ZipFile(_dst, "r")
        zf.extractall(os.path.dirname(_dst))
        zf.close()
        
        # change folder hierarchy
        inner_dir = os.path.join(root, 'kfold')
        for subdir in os.listdir(inner_dir):
            subdir = os.path.join(inner_dir, subdir)
            shutil.move(subdir, root)

        os.removedirs(inner_dir)


class PACSDataModule(SupervisedDataModule):
    def __init__(self,
                 root: str = 'data/domainbed/pacs',
                 train_environments: typing.Iterable[str] = ['P', 'A', 'C'],
                 test_environments: typing.Iterable[str] = ['S'],
                 validation_size: float = 0.2,
                 random_state: int = 42,
                 batch_size: int = 32,
                 num_workers: int = 1,
                 download: bool = False,
                 ):

        super().__init__()

        self.root = root
        self.download = download

        self.train_environments = train_environments
        self.test_environments = test_environments
        
        self.validation_size: float = validation_size
        self.random_state: int = random_state

        self.batch_size = batch_size
        self.num_workers = num_workers

        # collection of train / id-validation datasets
        self._train_datasets = []
        self._id_validation_datasets = []
        for env in self.train_environments:
            
            dataset = PACS(root=self.root, environments=[env])  # full training dataset
            
            tr_idx, val_idx = train_test_split(                 # split train / test indices
                np.arange(len(dataset)),
                test_size=self.validation_size,
                random_state=self.random_state,
                shuffle=True,
                stratify=dataset.labels  # 1d array
                )
            
            self._train_datasets.append(                        # subset by indices
                Subset(dataset, indices=torch.from_numpy(tr_idx))
            )
            
            self._id_validation_datasets.append(                # subset by indices
                Subset(dataset, indices=torch.from_numpy(val_idx))
            )

        # collection of test datasets
        self._test_datasets = [
            PACS(root=self.root, environments=[env]) for env in self.test_environments
        ]

    def prepare_data(self):
        """Download & extract data."""
        if self.download:
            PACS.download(root=self.root)

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

class PACSDep(MultipleDomainCollection):

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
        
        super().__init__()
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
