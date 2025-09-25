"""Run attribution on images and save visualizations."""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_interpretability.models import load_model
from vision_interpretability.models.preprocessing import preprocess_batch
from vision_interpretability.attribution import (
    VanillaGradients,
    IntegratedGradientsMethod,
    GradCAM,
)
from vision_interpretability.visualization import create_attribution_overlay


METHOD_MAP = {
    "vanilla_gradients": VanillaGradients,
    "integrated_gradients": IntegratedGradientsMethod,
    "gradcam": GradCAM,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run attribution methods on images"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (e.g., resnet50, vit_base_patch16_224)",
    )
    parser.add_argument(
        "--image-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input images",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=list(METHOD_MAP.keys()),
        default="integrated_gradients",
        help="Attribution method to use",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Target class (uses model prediction if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detected if not specified)",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Colormap for visualization",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha for overlay blending",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {args.model_name}")
    model, transform = load_model(args.model_name, device=args.device)
    device = next(model.parameters()).device
    
    print(f"Initializing attribution method: {args.method}")
    method_class = METHOD_MAP[args.method]
    
    if args.method == "gradcam":
        attribution_method = method_class(model, model_name=args.model_name)
    else:
        attribution_method = method_class(model)
    
    print(f"Processing {len(args.image_paths)} images...")
    
    for img_idx, img_path in enumerate(args.image_paths):
        print(f"\nProcessing: {img_path}")
        
        img_tensor = preprocess_batch([img_path], transform, device=str(device))
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
        
        target = args.target_class if args.target_class is not None else pred_class.item()
        
        print(f"  Predicted class: {pred_class.item()} (confidence: {confidence.item():.4f})")
        print(f"  Target class: {target}")
        print(f"  Computing attribution...")
        
        attribution = attribution_method.attribute(img_tensor, target)
        
        attr_np = attribution[0].detach().cpu().numpy()
        
        if attr_np.ndim == 3:
            attr_np = np.abs(attr_np).sum(axis=0)
        
        base_name = Path(img_path).stem
        
        np.save(output_dir / f"{base_name}_attribution.npy", attr_np)
        
        original_img = Image.open(img_path).convert("RGB")
        overlay = create_attribution_overlay(
            original_img,
            attr_np,
            alpha=args.alpha,
            colormap=args.colormap,
        )
        
        overlay.save(output_dir / f"{base_name}_overlay.png")
        
        print(f"  Saved: {base_name}_overlay.png")
    
    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
