"""Failure mode analysis utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image


@dataclass
class FailureExample:
    """Container for failure mode examples."""
    image_path: str
    true_label: int
    predicted_label: int
    confidence: float
    true_label_name: Optional[str] = None
    predicted_label_name: Optional[str] = None


def evaluate_model_on_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> dict:
    """
    Evaluate model on a dataset and collect predictions.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for dataset
        device: Device to run on
        max_batches: Maximum number of batches to process
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_paths = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            
            if len(batch) == 2:
                inputs, labels = batch
                paths = None
            else:
                inputs, labels, paths = batch
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            confidences, predictions = torch.max(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
            if paths is not None:
                all_paths.extend(paths)
            elif hasattr(dataloader.dataset, "get_path"):
                batch_start = batch_idx * dataloader.batch_size
                for i in range(len(labels)):
                    all_paths.append(dataloader.dataset.get_path(batch_start + i))
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    correct = all_predictions == all_labels
    accuracy = correct.mean()
    
    return {
        "predictions": all_predictions,
        "labels": all_labels,
        "confidences": all_confidences,
        "paths": all_paths,
        "correct": correct,
        "accuracy": accuracy,
    }


def find_high_confidence_misclassifications(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    threshold: float = 0.7,
    max_examples: Optional[int] = None,
    max_batches: Optional[int] = None,
) -> list[FailureExample]:
    """
    Find high-confidence misclassifications.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for dataset
        device: Device to run on
        threshold: Confidence threshold
        max_examples: Maximum number of examples to return
        max_batches: Maximum number of batches to process
        
    Returns:
        List of FailureExample objects
    """
    results = evaluate_model_on_dataset(model, dataloader, device, max_batches)
    
    failures = []
    
    for i in range(len(results["predictions"])):
        if not results["correct"][i] and results["confidences"][i] >= threshold:
            path = results["paths"][i] if i < len(results["paths"]) else f"index_{i}"
            
            failure = FailureExample(
                image_path=path,
                true_label=int(results["labels"][i]),
                predicted_label=int(results["predictions"][i]),
                confidence=float(results["confidences"][i]),
            )
            failures.append(failure)
            
            if max_examples and len(failures) >= max_examples:
                break
    
    failures.sort(key=lambda x: x.confidence, reverse=True)
    
    return failures


def find_class_confusion_pairs(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    top_k: int = 10,
    max_batches: Optional[int] = None,
) -> list[tuple[int, int, int]]:
    """
    Find the most common class confusion pairs.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for dataset
        device: Device to run on
        top_k: Number of top confusion pairs to return
        max_batches: Maximum number of batches to process
        
    Returns:
        List of (true_class, predicted_class, count) tuples
    """
    results = evaluate_model_on_dataset(model, dataloader, device, max_batches)
    
    confusion_counts = defaultdict(int)
    
    for i in range(len(results["predictions"])):
        if not results["correct"][i]:
            true_label = int(results["labels"][i])
            pred_label = int(results["predictions"][i])
            confusion_counts[(true_label, pred_label)] += 1
    
    sorted_confusions = sorted(
        confusion_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    return [(t, p, c) for (t, p), c in sorted_confusions[:top_k]]


def generate_failure_gallery(
    failures: list[FailureExample],
    attribution_method: Optional[Callable] = None,
    output_dir: Optional[str | Path] = None,
    max_examples: int = 20,
    ncols: int = 4,
) -> Image.Image:
    """
    Generate a gallery visualization of failure examples.
    
    Args:
        failures: List of FailureExample objects
        attribution_method: Optional attribution method to visualize
        output_dir: Optional directory to save individual images
        max_examples: Maximum number of examples to include
        ncols: Number of columns in grid
        
    Returns:
        PIL Image containing failure gallery
    """
    from ..visualization.grids import make_image_grid
    from ..visualization.overlay import create_attribution_overlay
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    gallery_images = []
    
    for idx, failure in enumerate(failures[:max_examples]):
        try:
            img = Image.open(failure.image_path).convert("RGB")
            
            if attribution_method is not None:
                import torch
                from ..models.preprocessing import preprocess_image
                
                img_tensor = preprocess_image(
                    failure.image_path,
                    attribution_method.model,
                )
                
                attribution = attribution_method.attribute(
                    img_tensor,
                    failure.predicted_label,
                )
                
                overlay = create_attribution_overlay(img, attribution[0])
                gallery_images.append(overlay)
            else:
                gallery_images.append(img)
            
            if output_dir:
                save_path = output_dir / f"failure_{idx:03d}.png"
                gallery_images[-1].save(save_path)
                
        except Exception as e:
            print(f"Error processing {failure.image_path}: {e}")
            continue
    
    if not gallery_images:
        return Image.new("RGB", (100, 100), (255, 255, 255))
    
    grid = make_image_grid(gallery_images, ncols=ncols)
    
    return grid


def save_failure_report(
    failures: list[FailureExample],
    output_path: str | Path,
    class_names: Optional[dict[int, str]] = None,
):
    """
    Save a CSV report of failure examples.
    
    Args:
        failures: List of FailureExample objects
        output_path: Path to save CSV file
        class_names: Optional mapping of class indices to names
    """
    import csv
    
    output_path = Path(output_path)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        writer.writerow([
            "image_path",
            "true_label",
            "predicted_label",
            "confidence",
            "true_label_name",
            "predicted_label_name",
        ])
        
        for failure in failures:
            true_name = class_names.get(failure.true_label, "") if class_names else ""
            pred_name = class_names.get(failure.predicted_label, "") if class_names else ""
            
            writer.writerow([
                failure.image_path,
                failure.true_label,
                failure.predicted_label,
                f"{failure.confidence:.4f}",
                true_name,
                pred_name,
            ])
