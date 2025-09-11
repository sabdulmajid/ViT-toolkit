"""Feature extraction using forward hooks."""

from typing import Any
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np


class FeatureExtractor:
    """Extract intermediate features from specified layers."""
    
    def __init__(self, model: nn.Module, layer_names: list[str]):
        """
        Args:
            model: PyTorch model
            layer_names: List of layer names to extract features from
        """
        self.model = model.eval()
        self.layer_names = layer_names
        self.features = OrderedDict()
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        def get_hook(name: str):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(get_hook(name))
                self.hooks.append(handle)
    
    def extract(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extract features for given inputs.
        
        Args:
            inputs: Input tensor of shape (N, C, H, W)
            
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        self.features.clear()
        
        with torch.no_grad():
            _ = self.model(inputs)
        
        return dict(self.features)
    
    def extract_numpy(self, inputs: torch.Tensor) -> dict[str, np.ndarray]:
        """
        Extract features as numpy arrays.
        
        Args:
            inputs: Input tensor of shape (N, C, H, W)
            
        Returns:
            Dictionary mapping layer names to feature arrays
        """
        features = self.extract(inputs)
        return {name: feat.cpu().numpy() for name, feat in features.items()}
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


def find_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module | None:
    """
    Find a layer in the model by its name.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer
        
    Returns:
        Layer module or None if not found
    """
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None


def get_layer_names(model: nn.Module, layer_type: type | None = None) -> list[str]:
    """
    Get all layer names in a model, optionally filtered by type.
    
    Args:
        model: PyTorch model
        layer_type: Optional layer type to filter by (e.g., nn.Conv2d)
        
    Returns:
        List of layer names
    """
    if layer_type is None:
        return [name for name, _ in model.named_modules() if name]
    else:
        return [name for name, module in model.named_modules() 
                if isinstance(module, layer_type) and name]
