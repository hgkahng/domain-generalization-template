
import torch
import torch.nn as nn
import torchvision.transforms as T


class DomainBedTransform(nn.Module):
    """Input transforms for DomainBed datasets."""

    _target_resolution = (224, 224)
    def __init__(self,
                 mean: tuple = (0.485, 0.456, 0.406),  # from ImageNet
                 std: tuple = (0.229, 0.224, 0.225),   # from ImageNet
                 augmentation: bool = False,
                 **kwargs):
        
        super().__init__()

        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        
        if self.augmentation:
            _transform: list = [
                T.Resize(self._target_resolution),
                T.RandomResizedCrop(self._target_resolution[0], scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.3, 0.3, 0.3, 0.3),
                T.RandomGrayscale(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(self.mean, self.std),
            ]
        else:
            _transform: list = [
                T.Resize(self._target_resolution),
                T.ConvertImageDtype(torch.float),
                T.Normalize(self.mean, self.std),
            ]

        self.transform = nn.Sequential(*_transform)

    def forward(self, x: torch.ByteTensor) -> torch.FloatTensor:
        return self.transform(x)
