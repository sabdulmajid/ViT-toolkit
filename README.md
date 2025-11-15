# Vision Interpretability Toolkit

Production-ready toolkit for interpreting pretrained vision models (ResNet, ConvNeXt, ViT) using PyTorch and Captum. Supports attribution methods, concept activation vectors, and failure mode analysis for computer vision research.

## Features

- **Attribution Methods**: Vanilla Gradients, Integrated Gradients, Grad-CAM, Guided Grad-CAM
- **Concept-Based Interpretability**: CAV (Concept Activation Vectors) with scikit-learn integration
- **Vision Transformer Support**: Full ViT support with attention pattern analysis
- **Failure Mode Analysis**: Automated detection of high-confidence misclassifications
- **Dataset Loaders**: ImageNet and Open Images with customizable preprocessing
- **GPU Efficient**: Optimized for consumer GPUs (tested on RTX 3060, 12GB VRAM)

## Installation

```bash
git clone https://github.com/sabdulmajid/ViT-toolkit.git
cd ViT-toolkit
pip install -r requirements.txt
pip install -e .
```

**Requirements:**
- Python 3.10+
- PyTorch 2.1+ (CUDA recommended)
- timm >=0.9.12
- Captum >=0.7.0

## Quick Start

### One-Command Case Study

Run a complete ViT-Base attribution analysis on sample images:

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File run_case_study.ps1
```

**Mac/Linux:**
```bash
chmod +x run_case_study.sh
./run_case_study.sh
```

Automatically downloads images, runs 3 attribution methods, and saves visualizations to `vit_base_patch16_224_case_study/`.

### Python API

```python
from vision_interpretability.models import load_model
from vision_interpretability.attribution import IntegratedGradientsMethod
from vision_interpretability.visualization import create_attribution_overlay

# Load pretrained model
model, transform = load_model("vit_base_patch16_224")

# Preprocess image
from vision_interpretability.models.preprocessing import preprocess_image
img_tensor = preprocess_image("image.jpg", transform)

# Compute attribution
ig = IntegratedGradientsMethod(model)
attribution = ig.attribute(img_tensor, target_class=282)

# Visualize
overlay = create_attribution_overlay("image.jpg", attribution[0])
overlay.save("result.png")
```

### CLI Scripts

**Run attribution on images:**
```bash
python scripts/run_attribution.py \
    --model-name vit_base_patch16_224 \
    --image-paths img1.jpg img2.jpg \
    --method integrated_gradients \
    --output-dir results/
```

**Generate failure mode gallery:**
```bash
python scripts/generate_failure_gallery.py \
    --model-name resnet50 \
    --dataset imagenet \
    --data-root /path/to/imagenet/val \
    --output-dir failure_analysis/ \
    --max-examples 50
```

**Extract CAV features:**
```bash
python scripts/extract_features_for_cav.py \
    --model-name resnet50 \
    --layer-name layer4 \
    --positive-list concept_positive.txt \
    --negative-list concept_negative.txt \
    --output-dir cav_output/
```

## Supported Models

| Model | Description | Parameters |
|-------|-------------|------------|
| `resnet50` | ResNet-50 | 25.6M |
| `convnext_base` | ConvNeXt Base | 88.6M |
| `vit_base_patch16_224` | Vision Transformer Base | 86.6M |
| `vit_small_patch16_224` | Vision Transformer Small | 22.1M |
| `efficientnet_b0` | EfficientNet B0 | 5.3M |

Extend model support in `vision_interpretability/models/registry.py`.

## Architecture

```
vision_interpretability/
├── models/          # Model loading and preprocessing
├── data/            # ImageNet and Open Images loaders
├── attribution/     # Captum-based attribution methods
├── concepts/        # CAV implementation and feature extraction
├── visualization/   # Overlay generation and grid utilities
└── eval/            # Failure mode detection
```

## Testing

```bash
pytest tests/ -v  # Run full test suite (9 tests)
```

All tests pass on CPU and CUDA devices.

## Case Study: ViT-Base Attribution Analysis

We evaluated ViT-Base (patch16_224) on 3 sample images using three attribution methods. Results demonstrate how different techniques highlight discriminative image regions.

### Results

| Image | Prediction | Confidence | Best Method |
|-------|------------|------------|-------------|
| Cat | Tabby cat (282) | 58.06% | Integrated Gradients |
| Dog | Weimaraner (156) | 87.45% | Grad-CAM |
| Goat | Goat (348) | 45.68% | Vanilla Gradients |

### Key Findings

**Method Characteristics:**
- **Vanilla Gradients**: Fine-grained, pixel-level attributions with high noise
- **Integrated Gradients**: Smooth attributions with superior semantic localization
- **Grad-CAM**: Coarse region-level heatmaps, interpretable for high-confidence predictions

**ViT Behavior Patterns:**
- High-confidence predictions (87.45%) yield focused, clean attribution maps
- Low-confidence predictions (45.68%) show diffuse attention across multiple regions
- Attention consistently focuses on semantically meaningful features (faces, limbs, characteristic traits)

**Practical Recommendations:**
- **Research**: Use Integrated Gradients for publication-quality visualizations
- **Debugging**: Apply Vanilla Gradients to identify spurious pixel dependencies
- **Communication**: Present Grad-CAM for intuitive, region-based explanations

**Reproduce this analysis:**
```bash
powershell -ExecutionPolicy Bypass -File run_case_study.ps1  # Windows
./run_case_study.sh  # Mac/Linux
```

Output: `vit_base_patch16_224_case_study/{vanilla_gradients,integrated_gradients,gradcam}/`

## Examples

Example notebooks in `notebooks/` (Python files with `# %%` cell markers):

1. `01_basic_resnet_attribution.py` - ResNet50 with multiple attribution methods
2. `02_vit_attribution_and_prisma.py` - Vision Transformer analysis with attention patterns
3. `03_failure_modes_gallery.py` - Automated failure detection and visualization

## License

MIT License - see [LICENSE](LICENSE) for details.
