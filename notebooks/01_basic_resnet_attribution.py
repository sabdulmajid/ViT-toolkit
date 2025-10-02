# ResNet Attribution - Notebook Guide
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
from vision_interpretability.attribution import VanillaGradients, IntegratedGradientsMethod, GradCAM
from vision_interpretability.visualization import create_attribution_overlay, make_image_grid

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Cell 2: Load Model
model, transform = load_model("resnet50", device=device)
print(f"Model loaded")

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

# Cell 5: Vanilla Gradients
vanilla_grad = VanillaGradients(model)
attribution_vg = vanilla_grad.attribute(img_tensor, pred.item())
overlay_vg = create_attribution_overlay(img_pil, attribution_vg[0], alpha=0.5)

plt.figure(figsize=(6, 6))
plt.imshow(overlay_vg)
plt.axis('off')
plt.title("Vanilla Gradients")
plt.show()

# Cell 6: Integrated Gradients
ig = IntegratedGradientsMethod(model, n_steps=50, baseline="black")
attribution_ig = ig.attribute(img_tensor, pred.item())
overlay_ig = create_attribution_overlay(img_pil, attribution_ig[0], alpha=0.5)

plt.figure(figsize=(6, 6))
plt.imshow(overlay_ig)
plt.axis('off')
plt.title("Integrated Gradients")
plt.show()

# Cell 7: Grad-CAM
gradcam = GradCAM(model, model_name="resnet50")
attribution_gc = gradcam.attribute(img_tensor, pred.item())
overlay_gc = create_attribution_overlay(img_pil, attribution_gc[0], alpha=0.5)

plt.figure(figsize=(6, 6))
plt.imshow(overlay_gc)
plt.axis('off')
plt.title("Grad-CAM")
plt.show()

# Cell 8: Comparison
images = [img_pil, overlay_vg, overlay_ig, overlay_gc]
grid = make_image_grid(images, ncols=4)

plt.figure(figsize=(16, 4))
plt.imshow(grid)
plt.axis('off')
plt.title("Original | Vanilla Gradients | Integrated Gradients | Grad-CAM")
plt.show()
