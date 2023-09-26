
import os
import glob
import typing
import pathlib
import shutil

import gdown

import numpy as np
import torch

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

from sklearn.model_selection import train_test_split

from dg_template.datasets.base import SupervisedDataModule
from dg_template.datasets.base import SemiSupervisedDataModule

from dg_template.datasets.utils import pil_loader
from dg_template.datasets.utils import train_test_split_by_n_samples_per_class


class OfficeHome(torch.utils.data.Dataset):
    _allowed_labels = (
        'Alarm_Clock',
        'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket',
        'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards',
        'Computer', 'Couch', 'Curtains',
        'Desk_Lamp', 'Drill',
        'Eraser', 'Exit_Sign',
        'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork',
        'Glasses',
        'Hammer', 'Helmet',
        'Kettle', 'Keyboard',
        'Knives',
        'Lamp_Shade', 'Laptop',
        'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug',
        'Notebook',
        'Oven',
        'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin',
        'Radio', 'Refrigerator', 'Ruler',
        'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers',
        'Soda', 'Speaker', 'Spoon',
        'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can',
        'Webcam'
    )
    _allowed_domains = ('Art', 'Clipart', 'Product', 'Real World')
    _size = (224, 224)
    _env_mapper = {
        'A': ('Art', 0),
        'C': ('Clipart', 1),
        'P': ('Product', 2),
        'R': ('Real World', 3),
    }
    
    def __init__(self,
                 root: str,
                 environments: typing.List[str] = ['A', 'C', 'P', 'R'],
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
            pattern = os.path.join(self.root, self._env_mapper[env][0], "**/*.jpg")
            files = glob.glob(pattern, recursive=True)
            self.input_files.extend(files)
        self.input_files = np.array(self.input_files)

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
        """Download & extract OfficeHome dataset."""
        
        url = "https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC"
        dst = "OfficeHome.zip"

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
        inner_dir = os.path.join(root, "OfficeHomeDataset_10072016")
        for subdir in os.listdir(inner_dir):
            subdir = os.path.join(inner_dir, subdir)
            shutil.move(subdir, root)

        os.removedirs(inner_dir)

class OfficeHomeDataModule(SupervisedDataModule):
    def __init__(self,
                 root: str = 'data/domainbed/officehome',
                 train_environments: typing.Iterable[str] = ['A', 'C', 'P'],
                 test_environments: typing.Iterable[str] = ['R'],
                 validation_size: float = 0.2,
                 random_state: int = 42,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 download: bool = False,
                 ):
        
        super().__init__()

        self.root: str = root
        self.download: bool = download

        self.train_environments = train_environments
        self.test_environments = test_environments

        self.validation_size: float = validation_size
        self.random_state: int = random_state

        self.batch_size = batch_size  # train batch size
        self.num_workers = num_workers  # number of cpu threads to use

        self._train_datasets = list()
        self._id_validation_datasets = list()
        for env in self.train_environments:

            dataset = OfficeHome(root=self.root, environments=[env])

            tr_idx, val_idx = train_test_split(
                np.arange(len(dataset)),
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=dataset.labels,
            )

            self._train_datasets.append(                        # subset by indices
                Subset(dataset, indices=torch.from_numpy(tr_idx))
            )
            
            self._id_validation_datasets.append(                # subset by indices
                Subset(dataset, indices=torch.from_numpy(val_idx))
            )

        self._test_datasets = [
            OfficeHome(root=self.root, environments=[env]) for env in self.test_environments
        ]

    def prepare_data(self):
        """Download & extract data."""
        if self.download:
            OfficeHome.download(root=self.root)

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
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
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
    

class SemiOfficeHomeDataModule(SemiSupervisedDataModule):
    def __init__(self,
                 root: str = 'data/domainbed/officehome',
                 train_environments: typing.Iterable[str] = ['A', 'C', 'P'],
                 test_environments: typing.Iterable[str] = ['R'],
                 validation_size: float = 0.2,
                 labeled_size: typing.Union[float, int] = 0.05,
                 random_state: int = 42,
                 batch_size: int = 32,
                 num_workers: int = 1,
                 download: bool = False,
                 ):

        super().__init__()

        self.root: str = root
        self.download: bool = download

        self.train_environments = train_environments
        self.test_environments = test_environments

        self.validation_size: float = validation_size
        if (self.validation_size <= 0.0) or (self.validation_size >= 1.0):
            raise ValueError(f"Invalid validation size: {self.validation_size}. Must be in (0.0, 1.0)")

        self.labeled_size: float = labeled_size  # TODO: support for exact number of examples per class
        assert self.labeled_size > 0

        self.random_state: int = random_state

        self.batch_size = batch_size
        self.num_workers = num_workers

        # collection of train (labeled & unlabeled) / id-validation datasets
        self._labeled_datasets = list()
        self._unlabeled_datasets = list()
        self._id_validation_datasets = list()

        for env in self.train_environments:

            dataset = OfficeHome(root=self.root, environments=[env])  # full training dataset

            # 1) split train / validation indices
            tr_idx, val_idx = train_test_split(
                np.arange(dataset.__len__()),
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=dataset.labels,
            )

            self._id_validation_datasets.append(
                Subset(dataset, indices=torch.from_numpy(val_idx))
            )

            # 2) split labeled & unlabeled indices
            if self.labeled_size < 1:
                
                # option a) stratified sampling perserving label distribution
                labeled_idx, unlabeled_idx = train_test_split(
                    tr_idx,
                    train_size=labeled_size,
                    random_state=random_state,
                    stratify=dataset.labels[tr_idx]
                )

            else:
                
                # option b) `self.labeled_size` specifies per-domain per-class number of samples
                n_samples_per_class_per_domain = int(self.labeled_size)
                labeled_idx, unlabeled_idx = train_test_split_by_n_samples_per_class(
                    tr_idx,
                    dataset.labels[tr_idx],
                    n_samples_per_class=n_samples_per_class_per_domain,
                    random_state=random_state,
                )

            self._labeled_datasets.append(
                Subset(dataset, indices=torch.from_numpy(labeled_idx))
            )

            self._unlabeled_datasets.append(
                Subset(dataset, indices=torch.from_numpy(unlabeled_idx))
            )

        # collection of test datasets
        self._test_datasets = [
            OfficeHome(root=self.root, environments=[env]) for env in self.test_environments
        ]

    def prepare_data(self):
        if self.download:
            OfficeHome.download(root=self.root)

    def setup(self, stage: str):
        
        if stage == 'fit':
            pass

        if stage == 'test':
            pass

        if stage == 'predict':
            pass

    def train_dataloader(self, **kwargs):
        return self._labeled_dataloader(**kwargs), self._unlabeled_dataloader(**kwargs)

    def _labeled_dataloader(self, **kwargs):
        concat = ConcatDataset(self._labeled_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers)
                          )

    def _unlabeled_dataloader(self, **kwargs):
        concat = ConcatDataset(self._unlabeled_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers)
                          )

    def val_dataloader(self, **kwargs):
        concat = ConcatDataset(self._id_validation_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers)
                          )

    def test_dataloader(self, **kwargs):
        concat = ConcatDataset(self._test_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          )