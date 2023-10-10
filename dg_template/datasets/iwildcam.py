
import os
import typing

import numpy as np
import pandas as pd
import torch

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import InterpolationMode, resize

from dg_template.datasets.base import SupervisedDataModule
from dg_template.datasets.base import SemiSupervisedDataModule
from dg_template.datasets.loaders import InfiniteDataLoader



class IWildCam(torch.utils.data.Dataset):
    """
        Input (x): RGB images from camera traps.
        Label (y): one of 186 classes corresponding to animal species.
            In the metadata, each instance is annotated with the ID of the location
            (camera trap) it came from.
    """
    
    _allowed_locations = list(range(323))
    _output_size = (448, 448)

    def __init__(self,
                 root: str = 'data/wilds/iwildcam_v2.0/',
                 locations: typing.Iterable[str] = [0],
                 split: typing.Optional[str] = None,
                 use_id_test: typing.Optional[bool] = True,
                 in_memory: typing.Optional[int] = 0,
                 ) -> None:
        super().__init__()

        self.root = root
        self.locations = locations
        self.split = split
        self.use_id_test = use_id_test
        self.in_memory = in_memory

        for l in self.locations:
            if l not in self._allowed_locations:
                raise ValueError(f'Invalid location: {l}')
            
        if self.split is not None:
            if self.split not in ('train', 'id_val'):
                raise ValueError(f'Invalid split: {self.split}')
            
        # Read metadata
        metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'), index_col=0)
        assert all([c in metadata.columns for c in ('split', 'location_remapped', 'y', 'filename')])

        # Keep rows of metadata specific to location(s) & split
        rows_to_keep = metadata['location_remapped'].isin(self.locations)
        if self.split == 'train':
            rows_to_keep = rows_to_keep & (metadata['split'] == 'train')
        elif self.split == 'id_val':
            if self.use_id_test:
                rows_to_keep = rows_to_keep & metadata['split'].isin(['id_val', 'id_test'])
            else:
                rows_to_keep = rows_to_keep & (metadata['split'] == 'id_val')
        
        metadata = metadata.loc[rows_to_keep, :].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)

        # Main attributes
        self.input_files = [
            os.path.join(self.root, 'train', filename) for filename in metadata['filename'].values
        ]  # all data instances are stored under the `train` folder
        if self.in_memory > 0:
            raise NotImplementedError("Work in progress...")
        else:
            self.inputs = None
        
        self.targets = torch.from_numpy(metadata['y'].values).long()
        self.domains = torch.from_numpy(metadata['location_remapped'].values).long()
        self.eval_groups = self.domains.clone()
        self.metadata = metadata

    def get_input(self, index: int) -> torch.ByteTensor:
        if self.inputs is not None:
            return self.inputs[index]
        else:
            img = read_image(self.input_files[index], mode=ImageReadMode.RGB)
            return resize(img=img, size=self._output_size,
                          interpolation=InterpolationMode.BILINEAR, antialias=True)
        
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
    


class UnlabeledIWildCam(torch.utils.data.Dataset):
    def __init__(self, root: str):
        super().__init__()


class IWildCamDataModule(SupervisedDataModule):
    def __init__(self, root: str):
        super().__init__()


class SemiIWildCamDataModule(SemiSupervisedDataModule):
    def __init__(self, root: str):
        super().__init__()
