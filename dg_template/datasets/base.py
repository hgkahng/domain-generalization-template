import os
import abc
import typing

import torch
import torch.nn as nn

from torch.utils.data import Dataset


class MultipleDomainDataset(torch.utils.data.Dataset):  # TODO: inherit torch.utils.Dataset ?
    def __init__(self):
        super().__init__()

        self.train_domains = list()
        self.validation_domains = list()
        self.test_domains = list()
        
        self._train_datasets = list()
        self._id_validation_datasets = list()
        self._ood_validation_datasets = list()  # some datasets do not have OOD validation sets
        self._test_datasets = list()

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_train_data(self, as_dict: bool = False) -> typing.Union[typing.Dict[str, Dataset],
                                                                    typing.List[Dataset],
                                                                    None]:
        if len(self._train_datasets) == 0:
            return None
        else:
            if as_dict:
                return {n: d for n, d in zip(self.train_domains, self._train_datasets)}
            return self._train_datasets

    def get_id_validation_data(self, as_dict: bool = False) -> typing.Union[typing.Dict[str, Dataset],
                                                                            typing.List[Dataset],
                                                                            None]:
        if len(self._id_validation_datasets) == 0:
            return None
        else:
            if as_dict:
                return {n: d for n, d in zip(self.train_domains, self._id_validation_dataset)}
            return self._id_validation_datasets

    def get_ood_validation_data(self, as_dict: bool = False) -> typing.Union[typing.Dict[str, Dataset],
                                                                             typing.List[Dataset],
                                                                             None]:
        if len(self._ood_validation_datasets) == 0:
            return None
        else:
            if as_dict:
                return {n: d for n, d in zip(self.validation_domains, self._ood_validation_datasets)}
            return self._ood_validation_datasets

    def get_test_data(self, as_dict: bool = False) -> typing.Union[typing.Dict[str, Dataset],
                                                                   typing.List[Dataset],
                                                                   None]:
        if len(self._test_datasets) == 0:
            return None
        else:
            if as_dict:
                return {n: d for n, d in zip(self.test_domains, self._test_datasets)}
            return self._test_datasets

    @property
    def input_shape(self) -> tuple:
        raise NotImplementedError
    
    @property
    def num_classes(self) -> int:
        raise NotImplementedError

    @staticmethod
    def as_list(x: typing.Union[int, typing.Iterable[int]]) -> typing.List[int]:
        if x is None:
            return []
        elif isinstance(x, int):
            return [x]
        elif isinstance(x, list):
            return x
        elif isinstance(x, tuple):
            return list(x)
        elif isinstance(x, dict):
            return [v for _, v in x.items()]
        else:
            raise NotImplementedError
