"""Quick smoke test to verify basic functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from vision_interpretability.models import load_model, list_available_models
        print("✓ Models module imported")
        
        from vision_interpretability.attribution import VanillaGradients, IntegratedGradientsMethod, GradCAM
        print("✓ Attribution module imported")
        
        from vision_interpretability.visualization import create_attribution_overlay, make_image_grid
        print("✓ Visualization module imported")
        
        from vision_interpretability.data import build_imagenet_dataloader
        print("✓ Data module imported")
        
        from vision_interpretability.concepts import FeatureExtractor
        print("✓ Concepts module imported")
        
        from vision_interpretability.eval import find_high_confidence_misclassifications
        print("✓ Eval module imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_registry():
    """Test model registry functionality."""
    print("\nTesting model registry...")
    
    try:
        from vision_interpretability.models import list_available_models, get_model_info
        
        models = list_available_models()
        print(f"✓ Found {len(models)} available models")
        
        info = get_model_info("resnet50")
        print(f"✓ Got model info: {info['name']}")
        
        return True
    except Exception as e:
        print(f"✗ Model registry test failed: {e}")
        return False


def test_model_loading():
    """Test loading a model (requires internet for download)."""
    print("\nTesting model loading...")
    
    try:
        import torch
        from vision_interpretability.models import load_model
        
        print("  Loading ResNet50 (this may take a while on first run)...")
        model, transform = load_model("resnet50", pretrained=True, device="cpu")
        
        print("✓ Model loaded successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False


def test_attribution():
    """Test attribution methods."""
    print("\nTesting attribution methods...")
    
    try:
        import torch
        from vision_interpretability.models import load_model
        from vision_interpretability.attribution import VanillaGradients
        
        model, _ = load_model("resnet50", pretrained=False, device="cpu")
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        
        vg = VanillaGradients(model)
        attribution = vg.attribute(dummy_input, target=0)
        
        print(f"✓ Attribution computed, shape: {attribution.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Attribution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Vision Interpretability Toolkit - Smoke Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Model Registry", test_model_registry()))
    
    # These tests require dependencies to be installed
    try:
        import torch
        results.append(("Model Loading", test_model_loading()))
        results.append(("Attribution", test_attribution()))
    except ImportError as e:
        print(f"\nSkipping model tests (missing dependencies): {e}")
        print("Run 'pip install -r requirements.txt' to install dependencies")
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. Check the output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
