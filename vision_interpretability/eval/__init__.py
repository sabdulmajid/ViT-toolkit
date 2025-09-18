"""Evaluation utilities for failure modes."""

from .failure_modes import (
    FailureExample,
    find_high_confidence_misclassifications,
    find_class_confusion_pairs,
    evaluate_model_on_dataset,
    generate_failure_gallery,
    save_failure_report,
)

__all__ = [
    "FailureExample",
    "find_high_confidence_misclassifications",
    "find_class_confusion_pairs",
    "evaluate_model_on_dataset",
    "generate_failure_gallery",
    "save_failure_report",
]
