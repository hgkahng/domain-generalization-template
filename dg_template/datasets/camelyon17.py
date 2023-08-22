
import os
import typing

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from torchvision.io import read_image, ImageReadMode

from dg_template.datasets.base import SupervisedDataModule
from dg_template.datasets.base import SemiSupervisedDataModule
from dg_template.datasets.loaders import InfiniteDataLoader


class Camelyon17(torch.utils.data.Dataset):

    _allowed_hospitals = (0, 1, 2, 3, 4)

    def __init__(self,
                 root: str = 'data/camelyon17_v1.0',
                 hospitals: typing.Iterable[int] = [0],
                 split: str = None,
                 in_memory: typing.Optional[int] = 0,
                 ) -> None:
        super().__init__()

        self.root = root
        self.hospitals = hospitals
        self.split = split
        self.in_memory = in_memory

        for h in self.hospitals:
            if h not in self._allowed_hospitals:
                raise ValueError
        
        if self.split is not None:
            if self.split not in ('train', 'val'):
                raise ValueError
            
        # Read metadata
        metadata = pd.read_csv(
            os.path.join(self.root, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'}
        )

        # Keep rows of metadata specific to hospital(s) & split
        rows_to_keep = metadata['center'].isin(hospitals)
        if self.split == 'train':
            rows_to_keep = rows_to_keep & (metadata['split'] == 0)
        elif self.split == 'val':
            rows_to_keep = rows_to_keep & (metadata['split'] == 1)

        metadata = metadata.loc[rows_to_keep].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)

        # Main attributes
        self.input_files = [
            os.path.join(
                self.root,
                f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            ) for patient, node, x, y in
            metadata.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)
        ]
        if self.in_memory > 0:
            raise NotImplementedError
        else:
            self.inputs = None
        
        self.targets = torch.LongTensor(metadata['tumor'].values)
        self.domains = torch.LongTensor(metadata['center'].values)
        self.eval_groups = torch.LongTensor(metadata['slide'].values)
        self.metadata = metadata

    def get_input(self, index: int) -> torch.ByteTensor:
        if self.input is not None:
            return self.inputs[index]
        else:
            return read_image(self.input_files[index], mode=ImageReadMode.RGB)
        
    def get_target(self, index: int) -> torch.LongTensor:
        return self.targets[index]
    
    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]
    
    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]
    
    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict(
            x=self.get_input(index),
            y=self.get_target(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    @staticmethod
    def download(root: str) -> None:
        raise NotImplementedError


class UnlabeledCamelyon17(torch.utils.data.Dataset):

    _allowed_hospitals = (0, 1, 2, 3, 4)

    def __init__(self,
                 root: str = 'data/camelyon17_unlabeled_v1.0',
                 hospitals: typing.Iterable[int] = [0],
                 ) -> None:
        
        super().__init__()

        self.root = root
        self.hospitals = hospitals

        for h in self.hospitals:
            if h not in self._allowed_hospitals:
                raise ValueError
            
        metadata = pd.read_csv(
            os.path.join(self.root, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'},
        )

        # keep rows of metadata specific to hospital(s) & split
        rows_to_keep = metadata['center'].isin(hospitals)
        metadata = metadata.loc[rows_to_keep].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)

        self.input_files = [
            os.path.join(
                self.root,
                f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            ) for patient, node, x, y in
            metadata.loc[:, ["patient", "node", "x_coord", "y_coord"]].itertuples(index=False, name=None)
        ]

        self.domains = torch.LongTensor(metadata['center'].values)
        self.eval_groups = torch.LongTensor(metadata['slide'].values)
        self.metadata = metadata

    def get_input(self, index: int) -> torch.ByteTensor:
        if self.input is not None:
            return self.inputs[index]
        else:
            return read_image(self.input_files[index], mode=ImageReadMode.RGB)
        
    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]
    
    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_grgoups[index]
    
    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict(
            x=self.get_input(index),
            y=self.get_target(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    @staticmethod
    def download(root: str) -> None:
        raise NotImplementedError


class Camelyon17DataModule(SupervisedDataModule):
    def __init__(self,
                 root: str = 'data/camelyon17_v1.0',
                 train_domains: typing.Iterable[int] = [0, 3, 4],
                 validation_domains: typing.Iterable[int] = [1],
                 test_domains: typing.Iterable[int] = [2],
                 batch_size: typing.Optional[int] = 32,
                 num_workers: typing.Optional[int] = 4,
                 prefetch_factor: typing.Optional[int] = 2,
                 ) -> None:
        
        super().__init__()

        self.root = root
        self.train_domains = [int(d) for d in train_domains] 
        self.validation_domains = [int(d) for d in validation_domains]
        self.test_domains = [int(d) for d in test_domains]
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self._train_datasets = []
        self._id_validation_datasets = []
        self._ood_validation_datasets = []
        self._id_test_datasets = []  # not supported yet
        self._ood_test_datasets = []

        for domain in self.train_domains:
            self._train_datasets += [
                Camelyon17(root=self.root, hospitals=[domain], split='train')
            ]
            self._id_validation_datasets += [
                Camelyon17(root=self.root, hospitals=[domain], split='val')
            ]

        for domain in self.validation_domains:
            self._ood_validation_datasets += [
                Camelyon17(root=self.root, hospitals=[domain], split=None)
            ]

        for domain in self.test_domains:
            self._ood_test_datasets += [
                Camelyon17(root=self.root, hospitals=[domain], split=None)
            ]

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):

        if stage == 'fit':
            pass

        if stage == 'test':
            pass

        if stage == 'predict':
            pass

    def train_dataloader(self, **kwargs):
        concat = ConcatDataset(self._train_datasets)
        if kwargs.get('infinite', False):
            loader_obj = InfiniteDataLoader
        else:
            loader_obj = DataLoader
        return loader_obj(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
    
    def val_dataloader(self, data_type: str = 'ood', **kwargs):
        if data_type == 'ood':
            return self._ood_val_dataloader(**kwargs)
        elif data_type == 'id':
            return self._id_val_dataloader(**kwargs)
        else:
            raise ValueError
        
    def test_dataloader(self, data_type: str = 'ood', **kwargs):
        if data_type == 'ood':
            return self._ood_test_dataloader(**kwargs)
        elif data_type == 'id':
            raise NotImplementedError
        else:
            raise ValueError
    
    def _id_val_dataloader(self, **kwargs):
        concat = ConcatDataset(self._id_validation_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
        
    def _ood_val_dataloader(self, **kwargs):
        concat = ConcatDataset(self._ood_validation_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
    
    def _id_test_dataloader(self, **kwargs):
        concat = ConcatDataset(self._id_test_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
        
    def _ood_test_dataloader(self, **kwargs):
        concat = ConcatDataset(self._test_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
    

class SemiCamelyon17DataModule(SemiSupervisedDataModule):
    def __init__(self,
                 root: str = 'data/camelyon17_v1.0',
                 unlabeled_root: str = 'data/camelyon17_unlabeled_v1.0',
                 train_domains: typing.Iterable[int] = [0, 3, 4],
                 validation_domains: typing.Iterable[int] = [1],
                 test_domains: typing.Iterable[int] = [2],
                 batch_size: typing.Optional[int] = 32,
                 num_workers: typing.Optional[int] = 4,
                 prefetch_factor: typing.Optional[int] = 2,
                 ):
        
        super().__init__()

        self.root: str = root
        self.unlabeled_root: str = unlabeled_root

        self.train_domains = [int(d) for d in train_domains] 
        self.validation_domains = [int(d) for d in validation_domains]
        self.test_domains = [int(d) for d in test_domains]

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.prefetch_factor: int = prefetch_factor

        self._labeled_train_datasets = []
        self._unlabeled_train_datasets = []
        self._id_validation_datasets = []
        self._ood_validation_datasets = []
        self._id_test_datasets = []  # not supported yet
        self._ood_test_datasets = []

        for domain in self.train_domains:
            
            # labeled train dataset
            self._labeled_train_datasets += [
                Camelyon17(root=self.root, hospitals=[domain], split='train')
            ]

            # unlabeled train dataset
            self._unlabeled_train_datasets += [
                UnlabeledCamelyon17(root=self.unlabeled_root, hospitals=[domain])
            ]

            # id validation dataset
            self._id_validation_datasets += [
                Camelyon17(root=self.root, hospitals=[domain], split='val')
            ]

        for domain in self.validation_domains:
            self._ood_validation_datasets += [
                Camelyon17(root=self.root, hospitals=[domain], split=None)
            ]

        for domain in self.test_domains:
            self._ood_test_datasets += [
                Camelyon17(root=self.root, hospitals=[domain], split=None)
            ]

    def prepare_data(self) -> None:
        pass

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
        concat = ConcatDataset(self._labeled_train_datasets)
        if kwargs.get('infinite', False):
            loader_obj = InfiniteDataLoader
        else:
            loader_obj = DataLoader
        return loader_obj(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
    
    def _unlabeled_dataloader(self, **kwargs):
        concat = ConcatDataset(self._unlabeled_train_datasets)
        if kwargs.get('infinite', False):
            loader_obj = InfiniteDataLoader
        else:
            loader_obj = DataLoader
        return loader_obj(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
    
    def val_dataloader(self, data_type: str = 'ood', **kwargs):
        if data_type == 'ood':
            return self._ood_val_dataloader(**kwargs)
        elif data_type == 'id':
            return self._id_val_dataloader(**kwargs)
        else:
            raise ValueError
        
    def test_dataloader(self, data_type: str = 'ood', **kwargs):
        if data_type == 'ood':
            return self._ood_test_dataloader(**kwargs)
        elif data_type == 'id':
            raise NotImplementedError
        else:
            raise ValueError
    
    def _id_val_dataloader(self, **kwargs):
        concat = ConcatDataset(self._id_validation_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
        
    def _ood_val_dataloader(self, **kwargs):
        concat = ConcatDataset(self._ood_validation_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
    
    def _id_test_dataloader(self, **kwargs):
        concat = ConcatDataset(self._id_test_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
        
    def _ood_test_dataloader(self, **kwargs):
        concat = ConcatDataset(self._test_datasets)
        return DataLoader(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
