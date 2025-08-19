"""Image preprocessing utilities."""

from pathlib import Path
from typing import Union, Optional, Callable

import torch
from PIL import Image
import numpy as np


def preprocess_image(
    image_path: Union[str, Path],
    transform: Callable,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to image file
        transform: torchvision transform to apply
        device: Target device for tensor
        
    Returns:
        Preprocessed image tensor of shape (1, C, H, W)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    return tensor


def preprocess_batch(
    image_paths: list[Union[str, Path]],
    transform: Callable,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Load and preprocess a batch of images.
    
    Args:
        image_paths: List of paths to image files
        transform: torchvision transform to apply
        device: Target device for tensor
        
    Returns:
        Batched tensor of shape (N, C, H, W)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tensors = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tensor = transform(image)
        tensors.append(tensor)
    
    batch = torch.stack(tensors).to(device)
    return batch


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Denormalize a tensor using ImageNet statistics or custom values.
    
    Args:
        tensor: Normalized tensor of shape (C, H, W) or (N, C, H, W)
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized tensor
    """
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.ndim == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)
    
    return tensor * std_t.to(tensor.device) + mean_t.to(tensor.device)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to a numpy image array.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (H, W)
        
    Returns:
        Numpy array of shape (H, W, C) or (H, W) with values in [0, 255]
    """
    if tensor.ndim == 3:
        img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    else:
        img = tensor.detach().cpu().numpy()
    
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img
