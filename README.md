# Neural Network From Scratch (MNIST)

## Overview

This project implements a **multi-layer feedforward neural network for MNIST digit classification** using **NumPy only** for the model: forward pass, backpropagation, and gradient descent. It demonstrates how classification networks work without deep-learning frameworks for the core training logic. The repository also includes **optional local inference UIs** (Flask and a static Netlify-ready site) that reuse the same trained weights.

## Key Features

- **Manual MLP**: 784 → 128 → 64 → 10 with ReLU hidden layers and softmax output
- **End-to-end training**: mini-batches, cross-entropy loss, He-style initialization, explicit backpropagation and SGD updates
- **Numerically stable softmax** and **clipped log** in cross-entropy for stable loss computation
- **Evaluation**: loss and accuracy on train/test splits, confusion matrix, optional matplotlib plots
- **Reproducibility**: configurable random seed; gradient sanity checks for debugging
- **Persistence**: save/load weights as `.npz`; export weights to JSON for browser inference
- **Data pipeline**: MNIST downloaded as gzip-compressed IDX files and parsed with the standard library plus NumPy (no TF/PyTorch for loading)
- **Deployment path**: static frontend in `site/` with `netlify.toml` publish directory

## Tech Stack

| Category | Technologies |
|----------|----------------|
| **Language / core ML** | Python 3, NumPy |
| **Training & evaluation** | Custom training loop (`train.py`), matplotlib (plots) |
| **Optional local web UI** | Flask (`webapp.py`) |
| **Static demo (Netlify)** | HTML, CSS, JavaScript (`site/`) |
| **Tooling** | Git, Netlify (static hosting via `site/` publish root) |

## Architecture / Approach

**Forward pass**: for batch \(X \in \mathbb{R}^{B \times 784}\), the network computes affine layers \(Z = XW + b\), applies ReLU on hidden pre-activations, and softmax on the final logits to obtain class probabilities.

**Loss**: categorical cross-entropy averaged over the batch.

**Backward pass**: gradients are derived with the chain rule. For softmax + cross-entropy, the gradient w.r.t. final logits uses the compact form \((P - Y)/B\), which avoids an explicit full Jacobian and keeps implementation clear. Hidden layers use the ReLU mask (derivative w.r.t. pre-activation).

**Design choices**: row-major batches `(batch, features)` for matrix multiplications; explicit shape assertions on gradients; bias gradients as sums over the batch dimension; optional numerical gradient check on a tiny batch for verification.

## Installation & Setup

```bash
git clone https://github.com/diaboliquepapillon/neural-network-from-scratch.git
cd neural-network-from-scratch

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

On first training run, MNIST files are downloaded into `data/mnist/` (ignored by git).

## Usage

**Train and evaluate** (saves `weights.npz` by default):

```bash
python train.py --epochs 15 --learning-rate 0.02 --batch-size 128 --seed 42 --no-visuals
```

**Debug** (first-batch shapes, optional gradient check):

```bash
python train.py --debug-shapes
python train.py --run-grad-check --epochs 1 --no-visuals
```

**Local Flask demo** (requires trained `weights.npz`):

```bash
python webapp.py
# Open http://127.0.0.1:5000
```

**Static / Netlify demo**: export weights for the browser, then deploy the `site/` directory.

```bash
python export_weights.py
# netlify.toml publishes site/
```

## Results / Output

- **Console**: per-epoch training and test loss and accuracy; final confusion matrix (counts per true vs. predicted class); path to saved `weights.npz`.
- **Optional plots** (when not using `--no-visuals`): training curves, sample prediction grid, confusion matrix heatmap.
- **Performance**: exact metrics depend on hyperparameters and seed. For this architecture on MNIST, accuracy typically improves markedly between early and later epochs; expect strong test performance after sufficient training relative to a single-epoch smoke run.

## Key Learnings

- Implementing **backprop by hand** clarifies tensor layouts, where averages enter the loss vs. gradients, and why **softmax + cross-entropy** simplifies the output-layer delta.
- **Numerical stability** (softmax shift, log clipping) is part of production-quality training loops, not an afterthought.
- **Shape discipline** (assertions, one documented layout) prevents subtle matrix-multiply bugs that frameworks usually hide.
- **Separating data I/O from the model** keeps the “from scratch” claim precise: the network is pure NumPy math; UIs are thin clients over saved weights.
- **Static deployment** for inference is viable when the forward pass is replicated in JavaScript and weights are exported once from NumPy.

## Future Improvements

- L2 weight decay and optional dropout for regularization experiments
- Momentum or Adam optimizer implemented in NumPy
- Learning-rate schedule and train/validation split with early stopping
- Automated tests for shapes and gradient-check tolerances
- Deeper or wider MLP baselines documented with comparable training budgets

---

**Repository**: [github.com/diaboliquepapillon/neural-network-from-scratch](https://github.com/diaboliquepapillon/neural-network-from-scratch)
