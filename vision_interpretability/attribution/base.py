"""Base class for attribution methods."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class AttributionMethod(ABC):
    """Base class for all attribution methods."""
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: PyTorch model to interpret
        """
        self.model = model.eval()
    
    @abstractmethod
    def attribute(
        self,
        inputs: torch.Tensor,
        target: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attribution map for inputs.
        
        Args:
            inputs: Input tensor of shape (N, C, H, W)
            target: Target class index (int) or tensor of shape (N,)
            
        Returns:
            Attribution map of shape (N, 1, H, W) or (N, H, W)
        """
        pass
