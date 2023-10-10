
import os
import typing

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from dg_template.datasets.base import SupervisedDataModule
from dg_template.datasets.base import SemiSupervisedDataModule
from dg_template.datasets.loaders import InfiniteDataLoader


class CivilComments(torch.utils.data.Dataset):

    _allowed_identities = [
        'male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white', 'not_mentioned',
    ]
    _auxiliary_vars = [
        'identity_any', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit',
    ]

    def __init__(self,
                 root: str = 'data/wilds/civilcomments_v1.0',
                 identity: str = 'male',
                 split: typing.Optional[str] = None,
                 model: typing.Optional[str] = 'distilbert-base-uncased',
                 metadata: typing.Optional[pd.DataFrame] = None,
                 ):
        super().__init__()

        self.root = root
        self.identity = identity
        self.split = split
        self.model = model

        # Set tokenizer
        if self.model is not None:
            self.tokenizer = self.get_bert_tokenizer(model=self.model)
        else:
            self.tokenizer = None
        
        if self.identity not in self._allowed_identities:
            raise ValueError(
                f"Invalid identity: {self.identity}. "
                f"Use one of [{', '.join(self._allowed_identities)}]."
            )
            
        if self.split is not None:
            if self.split not in ('train', 'val', 'test'):
                raise ValueError(
                    f"Invalid split: {self.split}. Use one of [train, val, test]."
                )
            
        # Read metadata (if not optionally provided as argument)
        # The metadata is quite large, so providing it from the init function
        # of the DataModule gives a little bit of speedup.
        if metadata is not None:
            assert isinstance(metadata, pd.DataFrame)
            metadata = metadata.copy()
        else:
            metadata = pd.read_csv(
                os.path.join(self.root, 'all_data_with_identities.csv'),
                index_col=0
            )
        
        # a dummy column for compatibility, replaced later.
        metadata['not_mentioned'] = 0
        metadata[self._allowed_identities] = (metadata[self._allowed_identities] >= 0.5).astype(int)
        metadata['not_mentioned'] = (metadata[self._allowed_identities].sum(axis=1) == 0).astype(int)

        # Keep rows relavant to current condition
        rows_to_keep = metadata[self.identity] > 0
        if self.split is not None:
            rows_to_keep = rows_to_keep & (metadata['split'] == self.split)
        metadata = metadata.loc[rows_to_keep, :].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)
        metadata['toxicity'] = (metadata['toxicity'] >= 0.5).astype(int)
        metadata['eval_group_str'] = metadata['toxicity'].apply(lambda x: f"{self.identity}_{x}")

        # Create a integer column for evaluation group.
        # Manually set `not_mentioned_0` and `not_mentioned_1` to -1,
        # so that it will be ignored in worst group metric calculation.
        _possible_eval_groups: list = self.possible_eval_groups
        metadata['eval_group'] = metadata['eval_group_str'].apply(lambda s: _possible_eval_groups.index(s))
        ignore_group_eval_mask = metadata['eval_group_str'].apply(lambda s: s.startswith('not_mentioned'))
        metadata.loc[ignore_group_eval_mask, 'eval_group'] = -1

        # Main attributes
        self.input_text = list(metadata['comment_text'])
        self.targets = torch.from_numpy(metadata['toxicity'].values > 0.5).long()
        self.domains = torch.LongTensor([self._allowed_identities.index(self.identity)] * len(self.targets))
        self.eval_groups = torch.from_numpy(metadata['eval_group'].values)

        self.metadata = metadata

    def get_input(self, index: int) -> torch.LongTensor:
        if self.tokenizer is not None:
            # Return as tensor
            tokens = self.tokenizer(
                self.input_text[index],
                padding='max_length',
                truncation=True,
                max_length=300,  # TODO: add to argument
                return_tensors='pt',
            )
            if self.model == 'bert-base-uncased':
                raise NotImplementedError
            elif self.model == 'distilbert-base-uncased':
                x = torch.stack([tokens['input_ids'], tokens['attention_mask']], dim=2)
            else:
                raise ValueError
            return torch.squeeze(x, dim=0)  # (max_length, 2) <- (1, max_length, 2)
        else:
            # Return as string text
            return self.input_text[index]

    def get_target(self, index: int) -> torch.LongTensor:
        return self.targets[index]

    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]

    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> typing.Dict[str, typing.Union[str, torch.Tensor]]:
        return dict(
            x=self.get_input(index),
            y=self.get_target(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )

    @property
    def possible_eval_groups(self) -> typing.List[str]:
        import itertools
        return [
            "_".join(tup) for tup in itertools.product(self._allowed_identities, ('0', '1'))
        ]

    @staticmethod
    def get_bert_tokenizer(model: str = 'distilbert-base-uncased'):
        """
        Implementation borrowed from:
            https://github.com/p-lambda/wilds/blob/f384c21c67ee58ab527d8868f6197e67c24764d4/examples/transforms.py#L88
        """
        if model == "bert-base-uncased":
            from transformers import BertTokenizerFast
            return BertTokenizerFast.from_pretrained(model)
        elif model == "distilbert-base-uncased":
            from transformers import DistilBertTokenizerFast
            return DistilBertTokenizerFast.from_pretrained(model)
        else:
            raise ValueError(f"Model: {model} not recognized.")
        

class UnlabeledCivilComments(torch.utils.data.Dataset):
    def __init__(self,
                 root: str = 'data/wilds/civilcomments_unlabeled_v1.0/',
                 model: typing.Optional[str] = 'distilbert-base-uncased',
                 ) -> None:
        super().__init__()

        self.root = root
        self.model = model
        
        # Set tokenizer
        if self.model is not None:
            self.tokenizer = CivilComments.get_bert_tokenizer(model=self.model)
        else:
            self.tokenizer = None

        # Read metadata
        metadata = pd.read_csv(
            os.path.join(self.root, 'unlabeled_data_with_identities.csv'),
            index_col=0
        )

        # Main attributes
        self.input_text = list(metadata['comment_text'])

        # I guess we actually have the labels (i.e., toxicity), but we are
        # just assuming we don't have them.
        self.targets = torch.LongTensor(metadata['toxicity'].values >= 0.5)
        self.domains = -1 * torch.ones_like(self.targets)
        self.eval_groups = -1 * torch.ones_like(self.targets)
        self.metadata = metadata

    def get_input(self, index: int) -> torch.LongTensor:
        if self.tokenizer is not None:
            # Return as tensor
            tokens = self.tokenizer(
                self.input_text[index],
                padding='max_length',
                truncation=True,
                max_length=300,  # TODO: add to argument
                return_tensors='pt',
            )
            if self.model == 'bert-base-uncased':
                raise NotImplementedError
            elif self.model == 'distilbert-base-uncased':
                x = torch.stack([tokens['input_ids'], tokens['attention_mask']], dim=2)
            else:
                raise ValueError
            return torch.squeeze(x, dim=0)  # (max_length, 2) <- (1, max_length, 2)
        else:
            # Return as string text
            return self.input_text[index]
        
    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]

    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]

    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, index: int) -> typing.Dict[str, typing.Union[str, torch.Tensor]]:
        return dict(
            x=self.get_input(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )


class CivilCommentsDataModule(SupervisedDataModule):
    def __init__(self,
                 root: str = 'data/wilds/civilcomments_v1.0/',  # TODO: check version
                 model: str = 'distilbert-base-uncased',
                 exclude_not_mentioned: bool = False,
                 batch_size: typing.Optional[int] = 32,
                 num_workers: typing.Optional[int] = 4,
                 prefetch_factor: typing.Optional[int] = 2,
                 ) -> None:
        super().__init__()

        self.root = root
        self.model = model
        self.exclude_not_mentioned = exclude_not_mentioned

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self._train_datasets = []
        self._validation_datasets = []
        self._test_datasets = []

        # domains across splits are equivalent
        identities: list = CivilComments._allowed_identities
        if exclude_not_mentioned:
            identities = [s for s in identities if s != 'not_mentioned']
        
        for identity in identities:
            
            # train dataset
            self._train_datasets.append(
                CivilComments(root=self.root, identity=identity,
                                split='train', model=self.model)
            )

            # validation dataset
            self._validation_datasets.append(
                CivilComments(root=self.root, identity=identity,
                                split='val', model=self.model)
            )
            
            # test dataset
            self._test_datasets.append(
                CivilComments(root=self.root, identity=identity,
                                split='test', model=self.model)
            )

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
        if kwargs.get('infinite', False):
            loader_obj = InfiniteDataLoader
        else:
            loader_obj = DataLoader
        return loader_obj(ConcatDataset(self._train_datasets),
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )

    def val_dataloader(self, **kwargs):
        return DataLoader(ConcatDataset(self._validation_datasets),
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
    
    def test_dataloader(self, **kwargs):
        return DataLoader(ConcatDataset(self._test_datasets),
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )


class CivilCommentsDGDataModule(SupervisedDataModule):
    def __init__(self,
                 root: str = 'data/wilds/civilcomments_v1.0/',
                 model: str = 'distilbert-base-uncased',
                 exclude_not_mentioned: bool = False,
                 train_domains: typing.Iterable[str] = [],       # TODO: discuss default values
                 validation_domains: typing.Iterable[str] = [],  # TODO: discuss default values
                 test_domains: typing.Iterable[str] = [],        # TODO: discuss default values
                 batch_size: typing.Optional[int] = 32,          # TODO: find optimal value
                 num_workers: typing.Optional[int] = 4,          # TODO: find optimal value
                 prefetch_factor: typing.Optional[int] = 2,      # TODO: find optimal value
                 ):
        super().__init__()

        self.root = root
        self.model = model
        self.exclude_not_mentioned = exclude_not_mentioned

        self.train_domains = train_domains
        self.validation_domains = validation_domains
        self.test_domains = test_domains

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self._train_datasets = []
        self._id_validation_datasets = []
        self._ood_validation_datasets = []
        self._id_test_datasets = []
        self._ood_test_datasets = []

        for domain in self.train_domains:
            
            self._train_datasets += [
                CivilComments(root=self.root, identity=domain, split='train', model=self.model)
            ]

            self._id_validation_datasets += [
                CivilComments(root=self.root, identity=domain, split='val', model=self.model)
            ]
            
            # TODO: should we include this in the training data
            self._id_test_datasets += [
                CivilComments(root=self.root, identity=domain, split='test', model=self.model)
            ]

        for domain in self.validation_domains:
            self._ood_validation_datasets += [
                CivilComments(root=self.root, identity=domain, split=None, model=self.model)
            ]

        for domain in self.test_domains:
            self._ood_test_datasets += [
                CivilComments(root=self.root, identity=domain, split=None, model=self.model)
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


class SemiCivilCommentsDataModule(SemiSupervisedDataModule):
    def __init__(self,
                 root: str = 'data/wilds/civilcomments_v1.0/',
                 unlabeled_root: str = 'data/wilds/civilcomments_unlabeled_v1.0/',
                 model: str = 'distilbert-base-uncased',
                 exclude_not_mentioned: bool = False,
                 batch_size: int = 32,
                 unlabeled_batch_size: int = 32,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 ) -> None:
        super().__init__()

        self.root = root
        self.unlabeled_root = unlabeled_root
        self.model = model
        self.exclude_not_mentioned = exclude_not_mentioned

        self._train_datasets = []
        self._validation_datasets = []
        self._test_datasets = []
        self._unlabeled_datasets = []

        # domains across splits are equivalent
        identities: list = CivilComments._allowed_identities
        if exclude_not_mentioned:
            identities = [s for s in identities if s != 'not_mentioned']

        for identity in identities:

            # train dataset
            self._train_datasets.append(
                CivilComments(root=self.root, identity=identity,
                                split='train', model=self.model)
            )

            # validation dataset
            self._validation_datasets.append(
                CivilComments(root=self.root, identity=identity,
                                split='val', model=self.model)
            )
            
            # test dataset
            self._test_datasets.append(
                CivilComments(root=self.root, identity=identity,
                                split='test', model=self.model)
            )
        
        # XXX: Unfortunately, unlabeled data does not have domains annotated.
        #      Some may belong to one of the eight training domains.
        self._unlabeled_datasets.append(
            UnlabeledCivilComments(root=self.unlabeled_root, model=self.model)
        )

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
    
    def _unlabeled_dataloader(self, **kwargs):
        concat = ConcatDataset(self._unlabeled_datasets)
        if kwargs.get('infinite', False):
            loader_obj = InfiniteDataLoader
        else:
            loader_obj = DataLoader
        return loader_obj(concat,
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
    
    def val_dataloader(self, **kwargs):
        return DataLoader(ConcatDataset(self._validation_datasets),
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )
    
    def test_dataloader(self, **kwargs):
        return DataLoader(ConcatDataset(self._test_datasets),
                          batch_size=kwargs.get('batch_size', self.batch_size),
                          num_workers=kwargs.get('num_workers', self.num_workers),
                          prefetch_factor=kwargs.get('prefetch_factor', self.prefetch_factor),
                          )


class SemiCivilCommentsDGDataModule(SemiSupervisedDataModule):
    # TODO:
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    # TODO: add usage example
    pass
