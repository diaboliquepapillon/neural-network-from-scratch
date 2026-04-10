"""Loss functions for classification."""

from __future__ import annotations

import numpy as np


def cross_entropy_loss(probs: np.ndarray, y_one_hot: np.ndarray, eps: float = 1e-12) -> float:
    """Compute average categorical cross-entropy loss.

    Args:
        probs: Predicted probabilities, shape (batch_size, num_classes)
        y_one_hot: One-hot encoded labels, shape (batch_size, num_classes)
    """
    assert probs.shape == y_one_hot.shape, "probs and y_one_hot must have same shape"
    clipped = np.clip(probs, eps, 1.0)
    batch_size = probs.shape[0]
    return -np.sum(y_one_hot * np.log(clipped)) / batch_size
