"""Model registry and loading utilities."""

from typing import Tuple, Optional
import warnings

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


SUPPORTED_MODELS = {
    "resnet50": "resnet50.a1_in1k",
    "convnext_base": "convnext_base.fb_in22k_ft_in1k",
    "vit_base_patch16_224": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "vit_small_patch16_224": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
}


def load_model(
    name: str,
    pretrained: bool = True,
    device: Optional[str] = None,
) -> Tuple[nn.Module, transforms.Compose]:
    """
    Load a timm model by name and return the model + preprocessing transform.
    
    Args:
        name: Model name (e.g., 'resnet50', 'vit_base_patch16_224')
        pretrained: Whether to load pretrained weights
        device: Target device ('cuda', 'cpu', or None for auto-detect)
        
    Returns:
        model: Model in eval mode on the requested device
        transform: Preprocessing transform for PIL images
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            warnings.warn("CUDA not available, using CPU. This will be slower.")
    
    if name not in SUPPORTED_MODELS:
        available = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(f"Model '{name}' not supported. Available: {available}")
    
    timm_name = SUPPORTED_MODELS[name]
    
    model = timm.create_model(timm_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    return model, transform


def get_model_info(name: str) -> dict:
    """
    Get metadata about a model without loading it.
    
    Args:
        name: Model name
        
    Returns:
        Dictionary with model metadata (input_size, num_classes, etc.)
    """
    if name not in SUPPORTED_MODELS:
        available = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(f"Model '{name}' not supported. Available: {available}")
    
    timm_name = SUPPORTED_MODELS[name]
    model = timm.create_model(timm_name, pretrained=False)
    config = resolve_data_config({}, model=model)
    
    return {
        "name": name,
        "timm_name": timm_name,
        "input_size": config["input_size"],
        "mean": config["mean"],
        "std": config["std"],
        "interpolation": config["interpolation"],
        "num_classes": model.num_classes if hasattr(model, "num_classes") else 1000,
    }


def list_available_models() -> list[str]:
    """Return list of supported model names."""
    return list(SUPPORTED_MODELS.keys())
