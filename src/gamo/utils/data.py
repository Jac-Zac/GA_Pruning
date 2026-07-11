import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from gamo.utils.paths import data_dir

_DATASET = datasets.FashionMNIST
FASHION_MNIST_NORMALIZATION = ((0.2860,), (0.3530,))


def _transforms():
    """FashionMNIST tensor conversion and normalization."""
    mean, std = FASHION_MNIST_NORMALIZATION
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def get_dataloaders(
    batch_size: int = 120,
    shuffle: bool = True,
    num_elements: int | None = None,
    val_split: float = 0.3,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load FashionMNIST and return train, validation, and test loaders.

    val_split: fraction of training data to use for validation (default 0.3).
    seed: optional seed for a deterministic train/validation split.
    """
    transform = _transforms()
    # Datasets are cached under data_dir() (artifacts/data by default, or wherever
    # ARTIFACTS_DIR points, e.g. the cluster scratch partition on HPC).
    root = data_dir()
    train_dataset = _DATASET(root=root, train=True, download=True, transform=transform)
    test_dataset = _DATASET(root=root, train=False, download=True, transform=transform)

    if num_elements is not None:
        train_dataset = Subset(
            train_dataset, range(min(num_elements, len(train_dataset)))
        )
        test_dataset = Subset(test_dataset, range(min(num_elements, len(test_dataset))))

    # Carve a validation split out of the training data. Train and val share the same
    # transform here, so a plain random_split is enough (no separate val dataset needed).
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
