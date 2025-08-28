"""Attribution overlay visualization."""

from typing import Optional

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def normalize_attribution(
    attribution: np.ndarray,
    percentile: float = 99,
) -> np.ndarray:
    """
    Normalize attribution map to [0, 1] range.
    
    Args:
        attribution: Attribution array of shape (H, W) or (C, H, W)
        percentile: Percentile for clipping outliers
        
    Returns:
        Normalized attribution of shape (H, W)
    """
    if attribution.ndim == 3:
        attribution = np.abs(attribution).sum(axis=0)
    
    vmax = np.percentile(attribution, percentile)
    vmin = attribution.min()
    
    attribution = np.clip(attribution, vmin, vmax)
    attribution = (attribution - vmin) / (vmax - vmin + 1e-8)
    
    return attribution


def apply_colormap(
    attribution: np.ndarray,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Apply colormap to attribution map.
    
    Args:
        attribution: Normalized attribution of shape (H, W)
        colormap: Matplotlib colormap name
        
    Returns:
        RGB image of shape (H, W, 3) with values in [0, 255]
    """
    cmap = cm.get_cmap(colormap)
    colored = cmap(attribution)
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return rgb


def create_attribution_overlay(
    image: np.ndarray | Image.Image | torch.Tensor,
    attribution: np.ndarray | torch.Tensor,
    alpha: float = 0.5,
    colormap: str = "jet",
    percentile: float = 99,
) -> Image.Image:
    """
    Create an overlay of attribution map on original image.
    
    Args:
        image: Original image (H, W, 3) or (C, H, W) or PIL Image
        attribution: Attribution map (H, W) or (C, H, W)
        alpha: Blending factor (0=only image, 1=only attribution)
        colormap: Matplotlib colormap name
        percentile: Percentile for normalizing attribution
        
    Returns:
        PIL Image with overlay
    """
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if isinstance(attribution, torch.Tensor):
        attribution = attribution.detach().cpu().numpy()
    
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    if image.shape[:2] != attribution.shape[-2:]:
        from PIL import Image as PILImage
        h, w = image.shape[:2]
        if attribution.ndim == 3:
            attribution = attribution.transpose(1, 2, 0)
        attr_img = PILImage.fromarray((attribution * 255).astype(np.uint8))
        attr_img = attr_img.resize((w, h), PILImage.BILINEAR)
        attribution = np.array(attr_img) / 255.0
    
    attribution_norm = normalize_attribution(attribution, percentile=percentile)
    
    attribution_colored = apply_colormap(attribution_norm, colormap=colormap)
    
    blended = (1 - alpha) * image + alpha * attribution_colored
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)


def create_side_by_side(
    image: np.ndarray | Image.Image,
    attribution: np.ndarray,
    colormap: str = "jet",
    percentile: float = 99,
) -> Image.Image:
    """
    Create side-by-side comparison of image and attribution.
    
    Args:
        image: Original image
        attribution: Attribution map
        colormap: Matplotlib colormap name
        percentile: Percentile for normalizing attribution
        
    Returns:
        PIL Image with side-by-side comparison
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    attribution_norm = normalize_attribution(attribution, percentile=percentile)
    attribution_colored = apply_colormap(attribution_norm, colormap=colormap)
    
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    combined = np.concatenate([image, attribution_colored], axis=1)
    return Image.fromarray(combined)
