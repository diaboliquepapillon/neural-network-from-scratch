# Neural Network From Scratch

A beginner-friendly project that implements a full feedforward neural network for MNIST digit classification using only:

- Python
- NumPy
- basic math
- matplotlib (for plots)

No TensorFlow/PyTorch/scikit-learn model APIs are used.  
MNIST is downloaded as raw IDX files and parsed manually with Python + NumPy.

---

## What "From Scratch" Means Here

This project manually implements:

- parameter initialization (`W`, `b`)
- linear layers (`Z = XW + b`)
- ReLU and Softmax activations
- cross-entropy loss
- backpropagation gradients for every layer
- gradient descent updates
- mini-batch training loop
- evaluation metrics (loss, accuracy, confusion matrix)
- save/load weights with NumPy (`.npz`)

---

## Concepts Covered

- Forward propagation through a multi-layer perceptron
- Backpropagation via chain rule
- Softmax + cross-entropy for multi-class classification
- Numerical stability tricks (stable softmax, clipped log)
- Batch averaging in gradients/loss
- Shape-aware debugging and assertions
- Optional numerical gradient checking

---

## Project Structure

```text
neural-network-from-scratch/
├── README.md
├── requirements.txt
├── data_loader.py
├── activations.py
├── losses.py
├── model.py
├── utils.py
├── visualize.py
├── train.py
├── webapp.py
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── app.js
```

---

## Network Architecture

- Input: `784` (flattened 28x28 image)
- Hidden layer 1: `128` + ReLU
- Hidden layer 2: `64` + ReLU
- Output: `10` + Softmax

---

## Math Walkthrough

### 1) Forward Propagation

For one mini-batch `X` with shape `(B, 784)`:

1. `Z1 = X @ W1 + b1` -> `(B, 128)`
2. `A1 = ReLU(Z1)` -> `(B, 128)`
3. `Z2 = A1 @ W2 + b2` -> `(B, 64)`
4. `A2 = ReLU(Z2)` -> `(B, 64)`
5. `Z3 = A2 @ W3 + b3` -> `(B, 10)`
6. `P = softmax(Z3)` -> `(B, 10)`

Each row of `P` is a probability distribution over digit classes `0..9`.

---

### 2) Why Softmax + Cross-Entropy Works Well

- Softmax converts raw scores (logits) into probabilities.
- Cross-entropy compares predicted probabilities to the true class distribution.
- Combined derivative becomes very clean:

`dL/dZ3 = (P - Y_one_hot) / B`

This avoids computing a large Jacobian manually and gives stable/efficient gradients.

---

### 3) Backpropagation (Layer by Layer)

Starting from output:

- `dZ3 = (P - Y) / B`
- `dW3 = A2^T @ dZ3`
- `db3 = sum(dZ3, axis=0, keepdims=True)`

Then hidden layer 2:

- `dA2 = dZ3 @ W3^T`
- `dZ2 = dA2 * ReLU'(Z2)`
- `dW2 = A1^T @ dZ2`
- `db2 = sum(dZ2, axis=0, keepdims=True)`

Then hidden layer 1:

- `dA1 = dZ2 @ W2^T`
- `dZ1 = dA1 * ReLU'(Z1)`
- `dW1 = X^T @ dZ1`
- `db1 = sum(dZ1, axis=0, keepdims=True)`

Bias gradient intuition: each neuron bias contributes equally to every sample in the batch, so gradients are summed over batch dimension.

---

### 4) Gradient Descent Updates

For each parameter `theta`:

`theta = theta - learning_rate * dtheta`

We do this for `W1, b1, W2, b2, W3, b3`.

---

### 5) Shape Safety

You will see:

- assertions in forward/backward
- gradient shape checks (`_assert_grad_shapes`)
- optional first-batch shape print (`--debug-shapes`)

This catches many silent bugs early.

---

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run Training

Basic run:

```bash
python train.py
```

With custom hyperparameters:

```bash
python train.py --epochs 15 --learning-rate 0.02 --batch-size 128 --seed 123
```

Enable shape debug (first batch):

```bash
python train.py --debug-shapes
```

Run numerical gradient check on a tiny mini-batch:

```bash
python train.py --run-grad-check --epochs 1 --no-visuals
```

Disable plots (useful on servers):

```bash
python train.py --no-visuals

Train first to produce `weights.npz` used by the website:

```bash
python train.py --epochs 10 --no-visuals
```

---

## Run the Website

Start the Flask app:

```bash
python webapp.py
```

Then open:

`http://127.0.0.1:5000`

You can draw a digit on the canvas and click **Predict**.
```

---

## Sample Output (Typical)

```text
Epoch 01/10 | train_loss=1.2254 train_acc=0.6551 | test_loss=0.5673 test_acc=0.8420
Epoch 02/10 | train_loss=0.4682 train_acc=0.8617 | test_loss=0.3995 test_acc=0.8859
...
Epoch 10/10 | train_loss=0.1840 train_acc=0.9465 | test_loss=0.2450 test_acc=0.9260
Saved weights to weights.npz
Confusion Matrix:
[[ 970    0    1 ...]
 ...
]
```

Exact numbers vary by seed, environment, and hardware.

---

## File-by-File Guide

- `data_loader.py`
  - Downloads MNIST IDX files on first run and parses them
  - Normalizes pixels to `[0, 1]`
  - Flattens images `28x28 -> 784`
  - One-hot encoding and mini-batch iterator

- `activations.py`
  - ReLU, ReLU derivative, numerically stable softmax

- `losses.py`
  - Categorical cross-entropy with clipping for stability

- `model.py`
  - Core neural network class
  - Forward pass, backpropagation, parameter updates
  - Evaluation + save/load (`np.savez`/`np.load`)

- `utils.py`
  - Seed setting, confusion matrix, shape debugging
  - Numerical gradient check helper for learning/debugging

- `visualize.py`
  - Training curves, prediction grid, confusion matrix plot

- `train.py`
  - CLI arguments
  - Training loop with mini-batches, metrics tracking
  - Final evaluation, optional visualizations, model saving

---

## Common Mistakes and Bugs

1. **Softmax overflow**  
   Fix: subtract per-row max before `exp`.

2. **Forgetting to average gradients by batch size**  
   Fix: use `dZ3 = (P - Y) / B`.

3. **Wrong matrix dimensions**  
   Fix: keep convention `(batch, features)` and check each `@` shape.

4. **Incorrect bias gradient**  
   Fix: `db = np.sum(dZ, axis=0, keepdims=True)`.

5. **Applying ReLU derivative to activated output instead of pre-activation**  
   Fix: use `ReLU'(Z)`, not `ReLU'(A)`.

6. **Numerical issues in cross-entropy**  
   Fix: clip probabilities before `log`.

---

## Possible Extensions

- Add L2 regularization / weight decay
- Add momentum or Adam optimizer (still from scratch)
- Implement dropout
- Add learning-rate scheduling
- Add validation split + early stopping
- Build a deeper network and compare convergence
- Implement convolutional layers from scratch (next challenge)

---

## Notes on Data Dependency

This project does not require TensorFlow or PyTorch.
MNIST data is fetched from a public dataset mirror and read from IDX format directly.
All neural network math and training logic are implemented manually in NumPy.
