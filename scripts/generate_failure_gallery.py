"""Generate failure mode gallery from dataset."""

import argparse
from pathlib import Path
import sys
import json

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_interpretability.models import load_model
from vision_interpretability.data import build_imagenet_dataloader, build_openimages_dataloader
from vision_interpretability.eval import (
    find_high_confidence_misclassifications,
    generate_failure_gallery,
    save_failure_report,
)
from vision_interpretability.attribution import IntegratedGradientsMethod


def main():
    parser = argparse.ArgumentParser(
        description="Generate failure mode gallery from dataset"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imagenet", "open_images"],
        required=True,
        help="Dataset type",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory for dataset",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="CSV path (for Open Images)",
    )
    parser.add_argument(
        "--class-names-file",
        type=str,
        default=None,
        help="JSON file with class index to name mapping",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Maximum number of failure examples",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for high-confidence misclassifications",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--use-attribution",
        action="store_true",
        help="Include attribution overlays in gallery",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = None
    if args.class_names_file:
        with open(args.class_names_file, "r") as f:
            class_names = json.load(f)
        if isinstance(next(iter(class_names.keys())), str):
            class_names = {int(k): v for k, v in class_names.items()}
    
    print(f"Loading model: {args.model_name}")
    model, transform = load_model(args.model_name, device=args.device)
    device = next(model.parameters()).device
    
    print(f"Building {args.dataset} dataloader...")
    if args.dataset == "imagenet":
        dataloader = build_imagenet_dataloader(
            root=args.data_root,
            transform=transform,
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        if not args.csv_path:
            raise ValueError("--csv-path required for Open Images dataset")
        
        dataloader = build_openimages_dataloader(
            csv_path=args.csv_path,
            image_dir=args.data_root,
            transform=transform,
            batch_size=args.batch_size,
            shuffle=False,
        )
    
    print("Finding high-confidence misclassifications...")
    failures = find_high_confidence_misclassifications(
        model=model,
        dataloader=dataloader,
        device=str(device),
        threshold=args.confidence_threshold,
        max_examples=args.max_examples,
        max_batches=args.max_batches,
    )
    
    print(f"Found {len(failures)} high-confidence misclassifications")
    
    if class_names:
        for failure in failures:
            failure.true_label_name = class_names.get(failure.true_label, "")
            failure.predicted_label_name = class_names.get(failure.predicted_label, "")
    
    csv_path = output_dir / "failures.csv"
    save_failure_report(failures, csv_path, class_names)
    print(f"Saved failure report to: {csv_path}")
    
    attribution_method = None
    if args.use_attribution:
        print("Initializing attribution method...")
        attribution_method = IntegratedGradientsMethod(model)
    
    print("Generating failure gallery...")
    gallery = generate_failure_gallery(
        failures=failures,
        attribution_method=attribution_method,
        output_dir=output_dir / "individual",
        max_examples=min(len(failures), args.max_examples),
        ncols=4,
    )
    
    gallery_path = output_dir / "failure_gallery.png"
    gallery.save(gallery_path)
    print(f"Saved gallery to: {gallery_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
