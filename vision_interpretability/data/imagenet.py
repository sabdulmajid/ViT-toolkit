"""ImageNet dataset utilities."""

from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


class ImageNetDataset(Dataset):
    """
    ImageNet dataset wrapper that supports class filtering.
    """
    
    def __init__(
        self,
        root: str,
        transform: Callable,
        class_indices: Optional[list[int]] = None,
    ):
        """
        Args:
            root: Root directory in ImageFolder format (root/class_name/image.jpg)
            transform: Transform to apply to images
            class_indices: Optional list of class indices to include (0-999 for ImageNet)
        """
        self.dataset = ImageFolder(root, transform=transform)
        self.class_indices = class_indices
        
        if class_indices is not None:
            class_set = set(class_indices)
            self.filtered_indices = [
                i for i, (_, label) in enumerate(self.dataset.samples)
                if label in class_set
            ]
        else:
            self.filtered_indices = list(range(len(self.dataset)))
    
    def __len__(self) -> int:
        return len(self.filtered_indices)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        real_idx = self.filtered_indices[idx]
        return self.dataset[real_idx]
    
    def get_path(self, idx: int) -> str:
        """Get the file path for a given index."""
        real_idx = self.filtered_indices[idx]
        return self.dataset.samples[real_idx][0]


def build_imagenet_dataloader(
    root: str,
    transform: Callable,
    class_names: Optional[list[str]] = None,
    class_indices: Optional[list[int]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for ImageNet.
    
    Args:
        root: Root directory in ImageFolder format
        transform: Transform to apply to images
        class_names: Optional list of class folder names to include
        class_indices: Optional list of class indices to include
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        pin_memory: Whether to use pinned memory (faster GPU transfer)
        
    Returns:
        DataLoader for ImageNet
    """
    if class_names is not None and class_indices is None:
        dataset_temp = ImageFolder(root)
        class_to_idx = dataset_temp.class_to_idx
        class_indices = [class_to_idx[name] for name in class_names if name in class_to_idx]
    
    dataset = ImageNetDataset(
        root=root,
        transform=transform,
        class_indices=class_indices,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return dataloader
