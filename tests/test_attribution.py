"""Tests for attribution methods."""

import pytest
import torch
import torch.nn as nn

from vision_interpretability.attribution import (
    VanillaGradients,
    IntegratedGradientsMethod,
    GradCAM,
)


class DummyModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def dummy_model():
    """Create a dummy model."""
    model = DummyModel()
    model.eval()
    return model


@pytest.fixture
def dummy_input():
    """Create dummy input."""
    return torch.randn(2, 3, 32, 32, requires_grad=True)


def test_vanilla_gradients(dummy_model, dummy_input):
    """Test vanilla gradients."""
    vg = VanillaGradients(dummy_model)
    attribution = vg.attribute(dummy_input, target=0)
    
    assert attribution.shape == dummy_input.shape
    assert not torch.isnan(attribution).any()
    assert not torch.isinf(attribution).any()


def test_integrated_gradients(dummy_model, dummy_input):
    """Test integrated gradients."""
    ig = IntegratedGradientsMethod(dummy_model, n_steps=10)
    attribution = ig.attribute(dummy_input, target=0)
    
    assert attribution.shape == dummy_input.shape
    assert not torch.isnan(attribution).any()
    assert not torch.isinf(attribution).any()


def test_gradcam(dummy_model, dummy_input):
    """Test Grad-CAM."""
    gc = GradCAM(dummy_model, target_layer=dummy_model.conv)
    attribution = gc.attribute(dummy_input, target=0)
    
    assert attribution.shape[0] == dummy_input.shape[0]
    assert attribution.shape[1] == 1
    assert attribution.shape[2] == dummy_input.shape[2]
    assert attribution.shape[3] == dummy_input.shape[3]
    assert not torch.isnan(attribution).any()
    assert not torch.isinf(attribution).any()
