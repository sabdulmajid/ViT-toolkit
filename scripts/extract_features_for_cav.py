"""Extract features for Concept Activation Vectors (CAVs)."""

import argparse
from pathlib import Path
import sys
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_interpretability.models import load_model
from vision_interpretability.concepts import collect_features_for_concept, train_cav


def load_image_list(file_path: str) -> list[str]:
    """Load list of image paths from text file."""
    with open(file_path, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract features and train CAV for a concept"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--layer-name",
        type=str,
        required=True,
        help="Layer name to extract features from",
    )
    parser.add_argument(
        "--positive-list",
        type=str,
        required=True,
        help="Text file with paths to positive concept images",
    )
    parser.add_argument(
        "--negative-list",
        type=str,
        required=True,
        help="Text file with paths to negative/random images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for CAV and features",
    )
    parser.add_argument(
        "--concept-name",
        type=str,
        default="concept",
        help="Name of the concept",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0,
        help="Regularization parameter C for logistic regression",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {args.model_name}")
    model, transform = load_model(args.model_name, device=args.device)
    device = next(model.parameters()).device
    
    print("Loading image lists...")
    positive_paths = load_image_list(args.positive_list)
    negative_paths = load_image_list(args.negative_list)
    
    print(f"Positive examples: {len(positive_paths)}")
    print(f"Negative examples: {len(negative_paths)}")
    
    print(f"\nExtracting features from layer: {args.layer_name}")
    X_pos, X_neg = collect_features_for_concept(
        model=model,
        layer_name=args.layer_name,
        positive_paths=positive_paths,
        negative_paths=negative_paths,
        transform=transform,
        batch_size=args.batch_size,
        device=str(device),
    )
    
    print(f"Positive features shape: {X_pos.shape}")
    print(f"Negative features shape: {X_neg.shape}")
    
    np.save(output_dir / "features_positive.npy", X_pos)
    np.save(output_dir / "features_negative.npy", X_neg)
    print("Saved feature arrays")
    
    print("\nTraining CAV...")
    cav, classifier, scaler = train_cav(
        X_pos=X_pos,
        X_neg=X_neg,
        C=args.regularization,
    )
    
    train_acc = classifier.score(
        scaler.transform(np.concatenate([X_pos, X_neg])),
        np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]),
    )
    
    print(f"CAV training accuracy: {train_acc:.4f}")
    
    np.save(output_dir / "cav.npy", cav)
    print(f"Saved CAV vector to: {output_dir / 'cav.npy'}")
    
    metadata = {
        "concept_name": args.concept_name,
        "model_name": args.model_name,
        "layer_name": args.layer_name,
        "n_positive": len(positive_paths),
        "n_negative": len(negative_paths),
        "cav_shape": cav.shape,
        "training_accuracy": float(train_acc),
        "regularization": args.regularization,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved metadata to: {output_dir / 'metadata.json'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
