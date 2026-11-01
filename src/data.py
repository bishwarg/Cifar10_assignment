
"""CIFAR-10 data preparation, augmentation, and data loaders."""
#import libraries
import random
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# CIFAR-10 channel-wise normalization values
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def seed_worker(worker_id):
    """Set reproducible random seeds for data-loading workers."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(
    batch_size=128,
    seed=42,
    data_dir="/content/CS6886W_VGG6_Assignment1/data",
    num_workers=2
):
    """Create training, validation, and test data loaders."""

    # Augmentation is applied only to training images.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Validation and test data must not receive random augmentation.
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Separate datasets allow different transformations for train and validation.
    train_aug = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    train_eval = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform
    )

    # Reproducible split: 45,000 training and 5,000 validation images.
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_aug), generator=generator).tolist()

    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_dataset = Subset(train_aug, train_indices)
    val_dataset = Subset(train_eval, val_indices)

    # Official CIFAR-10 test set: 10,000 images.
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform
    )

    loader_generator = torch.Generator().manual_seed(seed)

    common_args = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        **common_args
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_args
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_args
    )

    return train_loader, val_loader, test_loader
