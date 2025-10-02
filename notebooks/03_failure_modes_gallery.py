# Failure Modes Gallery - Notebook Guide
# Run these cells in sequence in a Jupyter notebook

# Cell 1: Imports
import sys
sys.path.insert(0, '..')

import torch
from pathlib import Path
import matplotlib.pyplot as plt

from vision_interpretability.models import load_model
from vision_interpretability.data import build_imagenet_dataloader
from vision_interpretability.eval import (
    find_high_confidence_misclassifications,
    find_class_confusion_pairs,
    generate_failure_gallery,
)
from vision_interpretability.attribution import IntegratedGradientsMethod

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Cell 2: Load Model
model, transform = load_model("resnet50", device=device)
print("Model loaded")

# Cell 3: Setup Dataset
# CHANGE THESE PATHS
data_root = "path/to/imagenet/val"  
batch_size = 32

dataloader = build_imagenet_dataloader(
    root=data_root,
    transform=transform,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)

print(f"Dataset ready with {len(dataloader)} batches")

# Cell 4: Find High-Confidence Misclassifications
print("Finding failures...")
failures = find_high_confidence_misclassifications(
    model=model,
    dataloader=dataloader,
    device=device,
    threshold=0.8,
    max_examples=50,
    max_batches=10,  # Process only first 10 batches for speed
)

print(f"Found {len(failures)} high-confidence misclassifications")

# Cell 5: Show Sample Failures
for i, failure in enumerate(failures[:5]):
    print(f"{i+1}. True: {failure.true_label}, Predicted: {failure.predicted_label}, "
          f"Confidence: {failure.confidence:.4f}")

# Cell 6: Find Class Confusion Pairs
confusions = find_class_confusion_pairs(
    model=model,
    dataloader=dataloader,
    device=device,
    top_k=10,
    max_batches=10,
)

print("\nTop class confusion pairs:")
for true_cls, pred_cls, count in confusions:
    print(f"  {true_cls} -> {pred_cls}: {count} times")

# Cell 7: Generate Failure Gallery (without attribution)
gallery = generate_failure_gallery(
    failures=failures[:20],
    attribution_method=None,
    max_examples=20,
    ncols=5,
)

plt.figure(figsize=(20, 12))
plt.imshow(gallery)
plt.axis('off')
plt.title("Failure Examples Gallery")
plt.show()

# Cell 8: Generate Failure Gallery with Attribution
print("Generating gallery with attribution (this may take a while)...")
ig = IntegratedGradientsMethod(model, n_steps=25)

gallery_with_attr = generate_failure_gallery(
    failures=failures[:12],
    attribution_method=ig,
    max_examples=12,
    ncols=4,
)

plt.figure(figsize=(16, 12))
plt.imshow(gallery_with_attr)
plt.axis('off')
plt.title("Failure Examples with Integrated Gradients")
plt.show()

# Cell 9: Save Results
output_dir = Path("failure_analysis_output")
output_dir.mkdir(exist_ok=True)

from vision_interpretability.eval import save_failure_report

save_failure_report(failures, output_dir / "failures.csv")
gallery.save(output_dir / "failure_gallery.png")
gallery_with_attr.save(output_dir / "failure_gallery_with_attribution.png")

print(f"\nResults saved to: {output_dir}")
