"""Tests for visualization utilities."""

import pytest
import numpy as np
from PIL import Image

from vision_interpretability.visualization import (
    create_attribution_overlay,
    make_image_grid,
)


def test_create_attribution_overlay():
    """Test creating attribution overlay."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    attribution = np.random.rand(100, 100)
    
    overlay = create_attribution_overlay(image, attribution, alpha=0.5)
    
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (100, 100)


def test_make_image_grid():
    """Test creating image grid."""
    images = [
        np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        for _ in range(6)
    ]
    
    grid = make_image_grid(images, ncols=3)
    
    assert isinstance(grid, Image.Image)
    assert grid.width > 50
    assert grid.height > 50
