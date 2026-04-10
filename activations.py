"""Activation functions used by the neural network."""

from __future__ import annotations

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU with respect to z."""
    return (z > 0).astype(z.dtype)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along class dimension.

    Args:
        logits: Shape (batch_size, num_classes)
    """
    # Subtract max(logits) per sample to avoid overflow in exp.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
