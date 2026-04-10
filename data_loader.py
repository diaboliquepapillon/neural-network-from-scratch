"""MNIST loading and preprocessing helpers.

This module downloads and parses the original IDX files directly
from public mirrors, so the project does not depend on TensorFlow
or PyTorch for data access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Tuple
from urllib.request import urlretrieve
import gzip
import struct

import numpy as np


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download_if_missing(path: Path, url: str) -> None:
    """Download file from URL only if it does not exist locally."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name} ...")
    urlretrieve(url, path.as_posix())  # nosec B310 (trusted public dataset URL)


def _read_idx_images(path: Path) -> np.ndarray:
    """Read an IDX image file and return shape (N, 28, 28)."""
    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_images, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """Read an IDX label file and return shape (N,)."""
    with gzip.open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_labels)


def load_mnist(data_dir: str = "data/mnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST from raw IDX files (downloaded automatically if needed).

    Returns:
        x_train: (60000, 784), float32 in [0, 1]
        y_train: (60000,), int64
        x_test: (10000, 784), float32 in [0, 1]
        y_test: (10000,), int64
    """
    root = Path(data_dir)
    paths = {
        "train_images": root / "train-images-idx3-ubyte.gz",
        "train_labels": root / "train-labels-idx1-ubyte.gz",
        "test_images": root / "t10k-images-idx3-ubyte.gz",
        "test_labels": root / "t10k-labels-idx1-ubyte.gz",
    }

    for key, url in MNIST_URLS.items():
        _download_if_missing(paths[key], url)

    x_train = _read_idx_images(paths["train_images"]).astype(np.float32) / 255.0
    x_test = _read_idx_images(paths["test_images"]).astype(np.float32) / 255.0
    y_train = _read_idx_labels(paths["train_labels"]).astype(np.int64)
    y_test = _read_idx_labels(paths["test_labels"]).astype(np.int64)

    # Flatten each 28x28 image into a vector of 784 features.
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, y_train, x_test, y_test


def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels to one-hot vectors."""
    out = np.zeros((y.size, num_classes), dtype=np.float32)
    out[np.arange(y.size), y] = 1.0
    return out


def batch_iterator(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield mini-batches from arrays x and y."""
    assert x.shape[0] == y.shape[0], "x and y must have same first dimension"
    n_samples = x.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]
