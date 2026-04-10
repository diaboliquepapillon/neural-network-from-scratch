"""Neural network implementation from scratch with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from activations import relu, relu_derivative, softmax
from losses import cross_entropy_loss


@dataclass
class Cache:
    """Intermediate values from forward pass for backpropagation."""

    x: np.ndarray
    z1: np.ndarray
    a1: np.ndarray
    z2: np.ndarray
    a2: np.ndarray
    z3: np.ndarray
    probs: np.ndarray


class NeuralNetwork:
    """Simple MLP: 784 -> 128 -> 64 -> 10."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        output_dim: int = 10,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        # He initialization is well suited for ReLU.
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim1)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim1), dtype=np.float32)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim1), size=(hidden_dim1, hidden_dim2)).astype(np.float32)
        self.b2 = np.zeros((1, hidden_dim2), dtype=np.float32)
        self.W3 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim2), size=(hidden_dim2, output_dim)).astype(np.float32)
        self.b3 = np.zeros((1, output_dim), dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Cache]:
        """Forward propagation.

        Shapes:
            x:     (B, 784)
            z1,a1: (B, 128)
            z2,a2: (B, 64)
            z3:    (B, 10)
            probs: (B, 10)
        """
        assert x.ndim == 2 and x.shape[1] == self.W1.shape[0], "x must have shape (batch, 784)"

        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        probs = softmax(z3)

        cache = Cache(x=x, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, probs=probs)
        return probs, cache

    def compute_loss(self, probs: np.ndarray, y_one_hot: np.ndarray) -> float:
        """Compute average cross-entropy loss."""
        return cross_entropy_loss(probs, y_one_hot)

    def backward(self, cache: Cache, y_one_hot: np.ndarray) -> Dict[str, np.ndarray]:
        """Backpropagation through the network.

        Key identity:
            If softmax + cross-entropy are combined, then:
            dL/dz3 = (probs - y_one_hot) / B
        where B is batch size.
        """
        x, z1, a1, z2, a2, probs = cache.x, cache.z1, cache.a1, cache.z2, cache.a2, cache.probs
        batch_size = x.shape[0]
        assert y_one_hot.shape == probs.shape, "y_one_hot must match probs shape"

        dz3 = (probs - y_one_hot) / batch_size
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_derivative(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
        self._assert_grad_shapes(grads)
        return grads

    def update_params(self, grads: Dict[str, np.ndarray], learning_rate: float) -> None:
        """Gradient descent parameter update."""
        self.W1 -= learning_rate * grads["dW1"]
        self.b1 -= learning_rate * grads["db1"]
        self.W2 -= learning_rate * grads["dW2"]
        self.b2 -= learning_rate * grads["db2"]
        self.W3 -= learning_rate * grads["dW3"]
        self.b3 -= learning_rate * grads["db3"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict integer class labels."""
        probs, _ = self.forward(x)
        return np.argmax(probs, axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 256) -> Tuple[float, float]:
        """Evaluate loss and accuracy."""
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        num_classes = self.b3.shape[1]
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            yb_one_hot = np.eye(num_classes, dtype=np.float32)[yb]
            probs, _ = self.forward(xb)
            total_loss += self.compute_loss(probs, yb_one_hot) * xb.shape[0]
            total_correct += np.sum(np.argmax(probs, axis=1) == yb)
            total_seen += xb.shape[0]
        return total_loss / total_seen, total_correct / total_seen

    def save(self, path: str) -> None:
        """Save model weights to a .npz file."""
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
        )

    def load(self, path: str) -> None:
        """Load model weights from a .npz file."""
        params = np.load(path)
        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]
        self.W3 = params["W3"]
        self.b3 = params["b3"]

    def _assert_grad_shapes(self, grads: Dict[str, np.ndarray]) -> None:
        assert grads["dW1"].shape == self.W1.shape
        assert grads["db1"].shape == self.b1.shape
        assert grads["dW2"].shape == self.W2.shape
        assert grads["db2"].shape == self.b2.shape
        assert grads["dW3"].shape == self.W3.shape
        assert grads["db3"].shape == self.b3.shape
