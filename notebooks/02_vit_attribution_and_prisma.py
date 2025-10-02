# ViT Attribution and Attention Analysis - Notebook Guide
# Run these cells in sequence in a Jupyter notebook

# Cell 1: Imports
import sys
sys.path.insert(0, '..')

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from vision_interpretability.models import load_model
from vision_interpretability.models.preprocessing import preprocess_image
from vision_interpretability.attribution import VanillaGradients, IntegratedGradientsMethod
from vision_interpretability.concepts import inspect_attention_patterns
from vision_interpretability.visualization import create_attribution_overlay

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Cell 2: Load ViT Model
model, transform = load_model("vit_base_patch16_224", device=device)
print(f"Model loaded: ViT Base")

# Cell 3: Load Sample Image
image_path = "path/to/your/image.jpg"  # CHANGE THIS
img_tensor = preprocess_image(image_path, transform, device=device)
img_pil = Image.open(image_path).convert("RGB")

plt.figure(figsize=(6, 6))
plt.imshow(img_pil)
plt.axis('off')
plt.title("Original Image")
plt.show()

# Cell 4: Get Prediction
with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, dim=1)

print(f"Predicted class: {pred.item()}, Confidence: {conf.item():.4f}")

# Cell 5: Vanilla Gradients for ViT
vanilla_grad = VanillaGradients(model)
attribution_vg = vanilla_grad.attribute(img_tensor, pred.item())
overlay_vg = create_attribution_overlay(img_pil, attribution_vg[0], alpha=0.5)

plt.figure(figsize=(6, 6))
plt.imshow(overlay_vg)
plt.axis('off')
plt.title("ViT Vanilla Gradients")
plt.show()

# Cell 6: Integrated Gradients for ViT
ig = IntegratedGradientsMethod(model, n_steps=50, baseline="black")
attribution_ig = ig.attribute(img_tensor, pred.item())
overlay_ig = create_attribution_overlay(img_pil, attribution_ig[0], alpha=0.5)

plt.figure(figsize=(6, 6))
plt.imshow(overlay_ig)
plt.axis('off')
plt.title("ViT Integrated Gradients")
plt.show()

# Cell 7: Inspect Attention Patterns
try:
    attention = inspect_attention_patterns(model, img_tensor, layer_idx=-1)
    if attention is not None:
        print(f"Attention shape: {attention.shape}")
        
        if attention.ndim == 4:
            att_avg = attention[0].mean(dim=0)
        else:
            att_avg = attention[0]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(att_avg.cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title("Average Attention Weights (Last Layer)")
        plt.show()
except Exception as e:
    print(f"Could not extract attention: {e}")

# Cell 8: Comparison
from vision_interpretability.visualization import make_image_grid

images = [img_pil, overlay_vg, overlay_ig]
grid = make_image_grid(images, ncols=3)

plt.figure(figsize=(15, 5))
plt.imshow(grid)
plt.axis('off')
plt.title("Original | Vanilla Gradients | Integrated Gradients")
plt.show()
