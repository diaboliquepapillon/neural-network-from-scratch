"""Visualization helpers for training and predictions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict) -> None:
    """Plot loss and accuracy curves from training history."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["test_loss"], label="test_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train_acc")
    axes[1].plot(epochs, history["test_acc"], label="test_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def show_predictions_grid(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_images: int = 25,
) -> None:
    """Display prediction samples in a square grid."""
    num_images = min(num_images, x.shape[0])
    side = int(np.sqrt(num_images))
    fig, axes = plt.subplots(side, side, figsize=(8, 8))
    axes = axes.ravel()
    for i in range(side * side):
        img = x[i].reshape(28, 28)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"T:{y_true[i]} P:{y_pred[i]}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray) -> None:
    """Plot a confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
