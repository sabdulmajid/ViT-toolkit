"""Concept Activation Vectors (CAVs) for concept-based interpretability."""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .feature_hooks import FeatureExtractor
from ..models.preprocessing import preprocess_batch


def collect_features_for_concept(
    model: nn.Module,
    layer_name: str,
    positive_paths: list[str | Path],
    negative_paths: list[str | Path],
    transform: Callable,
    batch_size: int = 32,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect features for positive and negative concept examples.
    
    Args:
        model: PyTorch model
        layer_name: Name of layer to extract features from
        positive_paths: Paths to positive concept examples
        negative_paths: Paths to negative/random examples
        transform: Preprocessing transform
        batch_size: Batch size for feature extraction
        device: Device to run on
        
    Returns:
        X_pos: Features for positive examples (N_pos, D)
        X_neg: Features for negative examples (N_neg, D)
    """
    extractor = FeatureExtractor(model, [layer_name])
    
    def extract_for_paths(paths: list[str | Path]) -> np.ndarray:
        all_features = []
        
        for i in tqdm(range(0, len(paths), batch_size), desc="Extracting features"):
            batch_paths = paths[i:i + batch_size]
            inputs = preprocess_batch(batch_paths, transform, device)
            
            features = extractor.extract_numpy(inputs)
            layer_features = features[layer_name]
            
            if layer_features.ndim == 4:
                layer_features = layer_features.mean(axis=(2, 3))
            elif layer_features.ndim > 2:
                layer_features = layer_features.reshape(layer_features.shape[0], -1)
            
            all_features.append(layer_features)
        
        return np.concatenate(all_features, axis=0)
    
    X_pos = extract_for_paths(positive_paths)
    X_neg = extract_for_paths(negative_paths)
    
    extractor.remove_hooks()
    
    return X_pos, X_neg


def train_cav(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> tuple[np.ndarray, LogisticRegression, StandardScaler]:
    """
    Train a Concept Activation Vector.
    
    Args:
        X_pos: Positive concept features (N_pos, D)
        X_neg: Negative concept features (N_neg, D)
        C: Regularization parameter
        max_iter: Maximum iterations for training
        
    Returns:
        cav: Normalized CAV vector (D,)
        classifier: Trained classifier
        scaler: Feature scaler
    """
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=42,
    )
    classifier.fit(X_scaled, y)
    
    cav = classifier.coef_[0]
    cav = cav / np.linalg.norm(cav)
    
    return cav, classifier, scaler


def concept_sensitivity(
    model: nn.Module,
    layer_name: str,
    cav: np.ndarray,
    inputs: torch.Tensor,
    target_class: int,
    scaler: StandardScaler | None = None,
) -> np.ndarray:
    """
    Compute concept sensitivity (directional derivative along CAV).
    
    Args:
        model: PyTorch model
        layer_name: Layer name where CAV was computed
        cav: CAV vector (D,)
        inputs: Input tensor (N, C, H, W)
        target_class: Target class for computing gradients
        scaler: Optional feature scaler
        
    Returns:
        Sensitivity scores (N,)
    """
    device = inputs.device
    cav_tensor = torch.tensor(cav, dtype=torch.float32, device=device)
    
    extractor = FeatureExtractor(model, [layer_name])
    
    inputs.requires_grad_(True)
    
    features_dict = extractor.extract(inputs)
    features = features_dict[layer_name]
    
    if features.ndim == 4:
        features = features.mean(dim=(2, 3))
    elif features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    
    if scaler is not None:
        features_np = features.detach().cpu().numpy()
        features_scaled = scaler.transform(features_np)
        features = torch.tensor(features_scaled, dtype=torch.float32, device=device)
    
    outputs = model(inputs)
    target_outputs = outputs[:, target_class]
    
    sensitivities = []
    for i in range(inputs.size(0)):
        grad_outputs = torch.zeros_like(features)
        grad_outputs[i] = cav_tensor
        
        grads = torch.autograd.grad(
            outputs=features,
            inputs=inputs,
            grad_outputs=grad_outputs,
            retain_graph=True,
        )[0]
        
        sensitivity = (grads[i] * inputs.grad if inputs.grad is not None else grads[i]).sum().item()
        sensitivities.append(sensitivity)
    
    extractor.remove_hooks()
    
    return np.array(sensitivities)
