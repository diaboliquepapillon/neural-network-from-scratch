"""Utility helpers: reproducibility, metrics, and gradient checking."""

from __future__ import annotations

from typing import Dict

import numpy as np


def set_seed(seed: int) -> None:
    """Set NumPy random seed for reproducibility."""
    np.random.seed(seed)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Compute confusion matrix with shape (num_classes, num_classes)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_shape_debug(batch_x: np.ndarray, probs: np.ndarray, grads: Dict[str, np.ndarray]) -> None:
    """Print useful shape info for debugging."""
    print("[Shape Debug]")
    print(f"  batch_x: {batch_x.shape}")
    print(f"  probs:   {probs.shape}")
    for name, value in grads.items():
        print(f"  {name}: {value.shape}")


def gradient_check_small_example(model, x_small: np.ndarray, y_small_one_hot: np.ndarray, epsilon: float = 1e-5) -> None:
    """Numerical gradient check for one entry in each parameter tensor.

    This is intentionally small and slow; use only for debugging and learning.
    """
    probs, cache = model.forward(x_small)
    analytic = model.backward(cache, y_small_one_hot)

    checks = [
        ("W1", "dW1", (0, 0)),
        ("b1", "db1", (0, 0)),
        ("W2", "dW2", (0, 0)),
        ("b2", "db2", (0, 0)),
        ("W3", "dW3", (0, 0)),
        ("b3", "db3", (0, 0)),
    ]
    print("[Gradient Check]")
    for param_name, grad_name, idx in checks:
        param = getattr(model, param_name)
        original = param[idx]

        param[idx] = original + epsilon
        probs_plus, _ = model.forward(x_small)
        loss_plus = model.compute_loss(probs_plus, y_small_one_hot)

        param[idx] = original - epsilon
        probs_minus, _ = model.forward(x_small)
        loss_minus = model.compute_loss(probs_minus, y_small_one_hot)

        param[idx] = original
        numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
        analytic_val = analytic[grad_name][idx]

        rel_error = abs(numerical - analytic_val) / max(1e-12, abs(numerical) + abs(analytic_val))
        print(
            f"  {param_name}{idx}: numerical={numerical:.6e}, "
            f"analytic={analytic_val:.6e}, rel_error={rel_error:.6e}"
        )
