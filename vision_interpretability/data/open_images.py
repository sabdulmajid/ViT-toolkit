"""Open Images dataset utilities."""

from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class OpenImagesSubsetDataset(Dataset):
    """
    Open Images subset dataset that works with a CSV of image IDs and labels.
    """
    
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Callable,
        image_id_col: str = "ImageID",
        label_col: str = "LabelName",
    ):
        """
        Args:
            csv_path: Path to CSV file with image IDs and labels
            image_dir: Directory containing image files
            transform: Transform to apply to images
            image_id_col: Name of column containing image IDs
            label_col: Name of column containing labels
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_id_col = image_id_col
        self.label_col = label_col
        
        unique_labels = self.df[label_col].unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id = row[self.image_id_col]
        label_name = row[self.label_col]
        
        image_path = self.image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = self.image_dir / f"{image_id}.png"
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        
        label_idx = self.label_to_idx[label_name]
        return image_tensor, label_idx
    
    def get_path(self, idx: int) -> str:
        """Get the file path for a given index."""
        row = self.df.iloc[idx]
        image_id = row[self.image_id_col]
        
        image_path = self.image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = self.image_dir / f"{image_id}.png"
        
        return str(image_path)


def build_openimages_dataloader(
    csv_path: str,
    image_dir: str,
    transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
    image_id_col: str = "ImageID",
    label_col: str = "LabelName",
) -> DataLoader:
    """
    Build a DataLoader for Open Images subset.
    
    Args:
        csv_path: Path to CSV file with image IDs and labels
        image_dir: Directory containing image files
        transform: Transform to apply to images
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        pin_memory: Whether to use pinned memory
        image_id_col: Name of column containing image IDs
        label_col: Name of column containing labels
        
    Returns:
        DataLoader for Open Images subset
    """
    dataset = OpenImagesSubsetDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        transform=transform,
        image_id_col=image_id_col,
        label_col=label_col,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return dataloader
