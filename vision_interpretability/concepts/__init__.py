"""Concept-based interpretability utilities."""

from .feature_hooks import FeatureExtractor
from .cav import collect_features_for_concept, train_cav, concept_sensitivity
from .prisma_bridge import load_vit_with_prisma, get_prisma_activations

__all__ = [
    "FeatureExtractor",
    "collect_features_for_concept",
    "train_cav",
    "concept_sensitivity",
    "load_vit_with_prisma",
    "get_prisma_activations",
]
