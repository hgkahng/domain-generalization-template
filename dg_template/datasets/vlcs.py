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
from torch.utils.data import Subset

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

from sklearn.model_selection import train_test_split

from dg_template.datasets.base import MultipleDomainCollection
from dg_template.datasets.base import MultipleDomainData
from dg_template.datasets.base import SupervisedDataModule
from dg_template.datasets.base import SemiSupervisedDataModule


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class VLCS(torch.utils.data.Dataset):

    """
    Folder hierarchy:
        - vlcs
            - Caltech101
            - LabelMe
            - SUN09
            - VOC2007
                - bird
                - car
                - chair
                - dog
                - person
    """

    _allowed_labels = ('bird', 'car', 'chair', 'dog', 'person')
    _allowed_domains = ('VOC2007', 'LabelMe', 'Caltech101', 'SUN09')
    _size = (224, 224)

    _url = "https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8"
    _env_mapper = {
        'V': ('VOC2007', 0),
        'L': ('LabelMe', 1),
        'C': ('Caltech101', 2),
        'S': ('SUN09', 3)
    }

    def __init__(self,
                 root: str = 'data/domainbed/vlcs/',
                 environments: typing.Iterable[str] = ['V', 'L', 'C', 'S'],
                 download: bool = False,
                 ) -> None:
        super().__init__()

        self.resize_fn = Resize(self._size)

        self.root = root
        self.environments = environments
        
        if download:
            self.download_and_extract(root)

        # collect input files (X)
        self.input_files = []
        for env in self.environments:
            pattern = os.path.join(self.root, self._env_mapper[env][0], "**/*.jpg")
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
    def download_and_extract(root: str) -> None:

        url = "https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8"
        dst = "VLCS.tar.gz"

        # download
        os.makedirs(root, exist_ok=True)
        _dst = os.path.join(root, dst)
        if not os.path.exists(_dst):
            gdown.download(url, _dst, quiet=False)
        
        # extract
        tar = tarfile.open(_dst, "r:gz")
        tar.extractall(root)
        tar.close()

        # change folder hierarchy
        inner_dir = os.path.join(root, 'VLCS')
        for subdir in os.listdir(inner_dir):
            subdir = os.path.join(inner_dir, subdir)
            shutil.move(subdir, root)

        os.removedirs(inner_dir)


class VLCSDataModule(SupervisedDataModule):
    """Add class docstring."""
    def __init__(self,
                 root: str,
                 train_environments: typing.Iterable[str] = ['V', 'L', 'C'],
                 test_environments: typing.Iterable[str] = ['S'],
                 validation_size: float = 0.2,
                 random_state: int = 42,
                 download: bool = False):
        
        super().__init__()
        
        self.root = root
        self.download = download

        self.train_environments = train_environments
        self.test_environments = test_environments
        
        self.validation_size: float = validation_size
        self.random_state: int = random_state

        # collection of train / id-validation datasets
        self._train_datasets = []
        self._id_validation_datasets = []
        for env in self.train_environments:
            
            dataset = VLCS(root=self.root, environments=[env])  # full training dataset
            
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
            VLCS(root=self.root, environments=[env]) for env in self.test_environments
        ]

    def prepare_data(self):
        """Download & extract data."""
        if self.download:
            VLCS.download_and_extract(root=self.root)

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


class VLCSDep(MultipleDomainCollection):
    
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
        
        super().__init__()
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
