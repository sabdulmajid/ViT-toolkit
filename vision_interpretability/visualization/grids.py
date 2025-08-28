"""Image grid visualization utilities."""

from typing import Optional

import numpy as np
from PIL import Image


def make_image_grid(
    images: list[np.ndarray | Image.Image],
    ncols: int,
    padding: int = 2,
    background: int = 255,
) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of images (numpy arrays or PIL Images)
        ncols: Number of columns
        padding: Padding between images in pixels
        background: Background color (0-255)
        
    Returns:
        PIL Image containing grid
    """
    pil_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            img = Image.fromarray(img)
        pil_images.append(img)
    
    if not pil_images:
        return Image.new("RGB", (100, 100), background)
    
    widths = [img.width for img in pil_images]
    heights = [img.height for img in pil_images]
    max_width = max(widths)
    max_height = max(heights)
    
    nrows = (len(pil_images) + ncols - 1) // ncols
    
    grid_width = ncols * max_width + (ncols + 1) * padding
    grid_height = nrows * max_height + (nrows + 1) * padding
    
    grid = Image.new("RGB", (grid_width, grid_height), background)
    
    for idx, img in enumerate(pil_images):
        row = idx // ncols
        col = idx % ncols
        
        x = col * max_width + (col + 1) * padding
        y = row * max_height + (row + 1) * padding
        
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2
        
        grid.paste(img, (x + x_offset, y + y_offset))
    
    return grid


def create_comparison_grid(
    images: list[np.ndarray | Image.Image],
    attributions: list[list[np.ndarray]],
    method_names: list[str],
    ncols: Optional[int] = None,
    overlay_alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """
    Create a comparison grid showing images with multiple attribution methods.
    
    Args:
        images: List of original images
        attributions: List of attribution lists (one per image, one per method)
        method_names: Names of attribution methods
        ncols: Number of columns (auto if None)
        overlay_alpha: Alpha for overlay blending
        colormap: Colormap for attributions
        
    Returns:
        PIL Image with comparison grid
    """
    from .overlay import create_attribution_overlay
    
    if ncols is None:
        ncols = len(method_names) + 1
    
    grid_images = []
    
    for img, attr_list in zip(images, attributions):
        grid_images.append(img)
        
        for attr in attr_list:
            overlay = create_attribution_overlay(
                img,
                attr,
                alpha=overlay_alpha,
                colormap=colormap,
            )
            grid_images.append(overlay)
    
    return make_image_grid(grid_images, ncols=ncols)


def add_labels_to_grid(
    grid: Image.Image,
    labels: list[str],
    ncols: int,
    font_size: int = 12,
) -> Image.Image:
    """
    Add text labels to a grid image.
    
    Args:
        grid: Grid image
        labels: List of labels (one per grid cell)
        ncols: Number of columns in grid
        font_size: Font size for labels
        
    Returns:
        Grid image with labels
    """
    from PIL import ImageDraw, ImageFont
    
    grid_copy = grid.copy()
    draw = ImageDraw.Draw(grid_copy)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    
    nrows = (len(labels) + ncols - 1) // ncols
    cell_width = grid.width // ncols
    cell_height = grid.height // nrows
    
    for idx, label in enumerate(labels):
        if idx >= ncols * nrows:
            break
        
        row = idx // ncols
        col = idx % ncols
        
        x = col * cell_width + 5
        y = row * cell_height + 5
        
        draw.text((x, y), label, fill=(255, 255, 255), font=font)
        draw.text((x + 1, y + 1), label, fill=(0, 0, 0), font=font)
    
    return grid_copy
