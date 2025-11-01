#!/bin/bash
# Vision Transformer Case Study - Mac/Linux with NVIDIA GPU
# Runs ViT-Base attribution analysis on sample images

set -e
echo "=== ViT Attribution Case Study ==="

# Setup directories
CASE_STUDY_DIR="case_study_output"
IMAGES_DIR="$CASE_STUDY_DIR/images"
mkdir -p "$IMAGES_DIR"

# Detect device
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "NVIDIA GPU detected - using CUDA"
else
    DEVICE="cpu"
    echo "No NVIDIA GPU detected - using CPU"
fi

# Download sample images (common ImageNet objects)
echo -e "\nDownloading sample images..."
declare -A IMAGES=(
    ["cat.jpg"]="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"
    ["dog.jpg"]="https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/640px-Cute_dog.jpg"
    ["goat.jpg"]="https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Domestic_goat_kid_in_capeweed.jpg/440px-Domestic_goat_kid_in_capeweed.jpg"
)

for name in "${!IMAGES[@]}"; do
    url="${IMAGES[$name]}"
    outpath="$IMAGES_DIR/$name"
    if [ ! -f "$outpath" ]; then
        echo "  - Downloading $name..."
        curl -sL "$url" -o "$outpath"
    fi
done

# Run attribution methods on all images
echo -e "\nRunning ViT-Base attribution analysis..."
IMAGE_PATHS=$(find "$IMAGES_DIR" -name "*.jpg" -type f)

for method in vanilla_gradients integrated_gradients gradcam; do
    echo "  - Running $method..."
    python scripts/run_attribution.py \
        --model-name vit_base_patch16_224 \
        --image-paths $IMAGE_PATHS \
        --method $method \
        --output-dir "$CASE_STUDY_DIR/$method" \
        --device $DEVICE
done

echo -e "\n=== Case Study Complete ==="
echo "Results saved to: $CASE_STUDY_DIR/"
echo -e "\nGenerated visualizations:"
find "$CASE_STUDY_DIR" -name "*.png" -type f | sed 's/^/  - /'
