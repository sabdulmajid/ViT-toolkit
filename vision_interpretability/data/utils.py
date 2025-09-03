"""Data utilities for class mapping and sampling."""

import json
from pathlib import Path
from typing import Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset


def load_class_mapping(json_path: str) -> dict[int, str]:
    """
    Load a mapping from class indices to human-readable labels.
    
    Args:
        json_path: Path to JSON file with mapping
        
    Returns:
        Dictionary mapping class index to label
    """
    with open(json_path, "r") as f:
        mapping = json.load(f)
    
    if isinstance(next(iter(mapping.keys())), str):
        mapping = {int(k): v for k, v in mapping.items()}
    
    return mapping


def sample_n_per_class(
    dataset: Dataset,
    n: int,
    max_classes: Optional[int] = None,
    seed: int = 42,
) -> list[int]:
    """
    Sample n examples from each class in a dataset.
    
    Args:
        dataset: Dataset with __getitem__ returning (image, label)
        n: Number of examples per class
        max_classes: Maximum number of classes to sample from
        seed: Random seed for reproducibility
        
    Returns:
        List of indices to sample
    """
    torch.manual_seed(seed)
    
    class_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_to_indices[label].append(idx)
    
    classes = sorted(class_to_indices.keys())
    if max_classes is not None:
        classes = classes[:max_classes]
    
    sampled_indices = []
    for cls in classes:
        indices = class_to_indices[cls]
        if len(indices) <= n:
            sampled_indices.extend(indices)
        else:
            perm = torch.randperm(len(indices))[:n]
            sampled_indices.extend([indices[i] for i in perm])
    
    return sampled_indices


def get_imagenet_classes() -> dict[int, str]:
    """
    Get ImageNet class index to name mapping.
    
    Returns:
        Dictionary mapping class index (0-999) to class name
    """
    try:
        import requests
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        response = requests.get(url)
        labels = response.json()
        return {i: label for i, label in enumerate(labels)}
    except Exception:
        return {i: f"class_{i}" for i in range(1000)}
