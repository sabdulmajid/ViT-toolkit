"""GradCAM attribution methods."""

import torch
import torch.nn as nn
from captum.attr import LayerGradCam, GuidedGradCam as CaptumGuidedGradCam

from .base import AttributionMethod


def get_target_layer(model: nn.Module, model_name: str = "") -> nn.Module:
    """
    Automatically select target layer for GradCAM based on model architecture.
    
    Args:
        model: PyTorch model
        model_name: Name of the model (e.g., 'resnet50', 'vit_base')
        
    Returns:
        Target layer module
    """
    model_name_lower = model_name.lower()
    
    if "resnet" in model_name_lower:
        return model.layer4[-1]
    elif "convnext" in model_name_lower:
        if hasattr(model, "stages"):
            return model.stages[-1]
        elif hasattr(model, "features"):
            return model.features[-1]
    elif "vit" in model_name_lower or "deit" in model_name_lower:
        if hasattr(model, "blocks"):
            return model.blocks[-1].norm1
        elif hasattr(model, "layers"):
            return model.layers[-1]
    elif "efficientnet" in model_name_lower:
        if hasattr(model, "conv_head"):
            return model.conv_head
        elif hasattr(model, "blocks"):
            return model.blocks[-1]
    
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module
    
    raise ValueError(f"Could not find suitable target layer for model: {model_name}")


class GradCAM(AttributionMethod):
    """Grad-CAM attribution method."""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        model_name: str = "",
    ):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to compute gradients for (auto-detected if None)
            model_name: Model name for auto layer selection
        """
        super().__init__(model)
        
        if target_layer is None:
            target_layer = get_target_layer(model, model_name)
        
        self.target_layer = target_layer
        self.gradcam = LayerGradCam(self.model, target_layer)
    
    def attribute(
        self,
        inputs: torch.Tensor,
        target: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Grad-CAM attribution.
        
        Args:
            inputs: Input tensor of shape (N, C, H, W)
            target: Target class index or tensor of shape (N,)
            
        Returns:
            Attribution map of shape (N, 1, H, W)
        """
        if isinstance(target, int):
            target = torch.tensor([target] * inputs.size(0), device=inputs.device)
        
        with torch.enable_grad():
            attributions = self.gradcam.attribute(inputs, target=target)
        
        if attributions.ndim == 3:
            attributions = attributions.unsqueeze(1)
        
        import torch.nn.functional as F
        attributions = F.interpolate(
            attributions,
            size=inputs.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        
        return attributions


class GuidedGradCAM(AttributionMethod):
    """Guided Grad-CAM attribution method."""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        model_name: str = "",
    ):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to compute gradients for (auto-detected if None)
            model_name: Model name for auto layer selection
        """
        super().__init__(model)
        
        if target_layer is None:
            target_layer = get_target_layer(model, model_name)
        
        self.target_layer = target_layer
        self.guided_gradcam = CaptumGuidedGradCam(self.model, target_layer)
    
    def attribute(
        self,
        inputs: torch.Tensor,
        target: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Guided Grad-CAM attribution.
        
        Args:
            inputs: Input tensor of shape (N, C, H, W)
            target: Target class index or tensor of shape (N,)
            
        Returns:
            Attribution map of shape (N, C, H, W)
        """
        if isinstance(target, int):
            target = torch.tensor([target] * inputs.size(0), device=inputs.device)
        
        with torch.enable_grad():
            attributions = self.guided_gradcam.attribute(inputs, target=target)
        
        return attributions
