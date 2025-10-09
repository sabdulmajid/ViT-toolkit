"""Tests for model loading and registry."""

import pytest
import torch

from vision_interpretability.models import load_model, get_model_info, list_available_models


def test_list_available_models():
    """Test that we can list available models."""
    models = list_available_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "resnet50" in models
    assert "vit_base_patch16_224" in models


def test_get_model_info():
    """Test model info retrieval."""
    info = get_model_info("resnet50")
    assert isinstance(info, dict)
    assert "name" in info
    assert "input_size" in info
    assert "num_classes" in info


@pytest.mark.parametrize("model_name", ["resnet50", "vit_base_patch16_224"])
def test_load_model(model_name):
    """Test model loading."""
    model, transform = load_model(model_name, pretrained=False, device="cpu")
    
    assert model is not None
    assert transform is not None
    assert not model.training
    
    batch_size = 2
    channels, height, width = 3, 224, 224
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape[0] == batch_size
    assert output.shape[1] == 1000  # ImageNet classes
