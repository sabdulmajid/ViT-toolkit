"""Saliency-based attribution methods."""

import torch
import torch.nn as nn
from captum.attr import Saliency, IntegratedGradients

from .base import AttributionMethod


class VanillaGradients(AttributionMethod):
    """Vanilla gradient-based saliency maps."""
    
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.saliency = Saliency(self.model)
    
    def attribute(
        self,
        inputs: torch.Tensor,
        target: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute vanilla gradient saliency map.
        
        Args:
            inputs: Input tensor of shape (N, C, H, W)
            target: Target class index or tensor of shape (N,)
            
        Returns:
            Attribution map of shape (N, C, H, W)
        """
        if isinstance(target, int):
            target = torch.tensor([target] * inputs.size(0), device=inputs.device)
        
        with torch.enable_grad():
            attributions = self.saliency.attribute(inputs, target=target)
        
        return attributions


class IntegratedGradientsMethod(AttributionMethod):
    """Integrated Gradients attribution method."""
    
    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        baseline: str = "black",
    ):
        """
        Args:
            model: PyTorch model
            n_steps: Number of integration steps
            baseline: Baseline type ('black', 'white', or 'blur')
        """
        super().__init__(model)
        self.ig = IntegratedGradients(self.model)
        self.n_steps = n_steps
        self.baseline_type = baseline
    
    def _get_baseline(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate baseline tensor."""
        if self.baseline_type == "black":
            return torch.zeros_like(inputs)
        elif self.baseline_type == "white":
            return torch.ones_like(inputs)
        elif self.baseline_type == "blur":
            import torch.nn.functional as F
            kernel_size = 15
            sigma = 5.0
            
            channels = inputs.size(1)
            kernel = torch.zeros(1, 1, kernel_size, kernel_size)
            center = kernel_size // 2
            
            for i in range(kernel_size):
                for j in range(kernel_size):
                    kernel[0, 0, i, j] = torch.exp(
                        -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2)
                    )
            kernel = kernel / kernel.sum()
            kernel = kernel.repeat(channels, 1, 1, 1).to(inputs.device)
            
            blurred = F.conv2d(inputs, kernel, padding=kernel_size // 2, groups=channels)
            return blurred
        else:
            return torch.zeros_like(inputs)
    
    def attribute(
        self,
        inputs: torch.Tensor,
        target: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            inputs: Input tensor of shape (N, C, H, W)
            target: Target class index or tensor of shape (N,)
            
        Returns:
            Attribution map of shape (N, C, H, W)
        """
        if isinstance(target, int):
            target = torch.tensor([target] * inputs.size(0), device=inputs.device)
        
        baselines = self._get_baseline(inputs)
        
        with torch.enable_grad():
            attributions = self.ig.attribute(
                inputs,
                baselines=baselines,
                target=target,
                n_steps=self.n_steps,
            )
        
        return attributions
