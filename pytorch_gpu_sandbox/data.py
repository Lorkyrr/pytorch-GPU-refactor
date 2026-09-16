"""Carregamento do CIFAR-10 (download automático + transforms + DataLoaders)."""

from __future__ import annotations

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .constants import CIFAR10_DESVIO_PADRAO, CIFAR10_MEDIA, CIFAR10_NUM_WORKERS_PADRAO, CIFAR10_RANDOM_CROP_PADDING


def cifar10_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = CIFAR10_NUM_WORKERS_PADRAO,
) -> tuple[DataLoader, DataLoader]:
    transform_treino = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=CIFAR10_RANDOM_CROP_PADDING),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEDIA, CIFAR10_DESVIO_PADRAO),
        ]
    )
    transform_teste = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEDIA, CIFAR10_DESVIO_PADRAO),
        ]
    )

    conjunto_treino = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_treino)
    conjunto_teste = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_teste)

    loader_treino = DataLoader(
        conjunto_treino, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    loader_teste = DataLoader(
        conjunto_teste, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return loader_treino, loader_teste
