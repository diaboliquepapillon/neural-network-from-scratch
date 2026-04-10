"""Train and evaluate a NumPy neural network on MNIST."""

from __future__ import annotations

import argparse

import numpy as np

from data_loader import batch_iterator, load_mnist, one_hot_encode
from model import NeuralNetwork
from utils import confusion_matrix, gradient_check_small_example, print_shape_debug, set_seed
from visualize import plot_confusion_matrix, plot_training_curves, show_predictions_grid


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Neural Network From Scratch - MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-shapes", action="store_true")
    parser.add_argument("--run-grad-check", action="store_true")
    parser.add_argument("--save-path", type=str, default="weights.npz")
    parser.add_argument("--no-visuals", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Main training workflow."""
    set_seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist()

    model = NeuralNetwork(seed=args.seed)
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    if args.run_grad_check:
        x_small = x_train[:4]
        y_small = one_hot_encode(y_train[:4], num_classes=10)
        gradient_check_small_example(model, x_small, y_small)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        samples_seen = 0

        iterator = batch_iterator(
            x_train, y_train, batch_size=args.batch_size, shuffle=True, seed=args.seed + epoch
        )
        for step, (xb, yb) in enumerate(iterator):
            yb_one_hot = one_hot_encode(yb, num_classes=10)

            probs, cache = model.forward(xb)
            loss = model.compute_loss(probs, yb_one_hot)
            grads = model.backward(cache, yb_one_hot)
            model.update_params(grads, learning_rate=args.learning_rate)

            if args.debug_shapes and epoch == 1 and step == 0:
                print_shape_debug(xb, probs, grads)

            batch_size = xb.shape[0]
            epoch_loss += loss * batch_size
            epoch_correct += np.sum(np.argmax(probs, axis=1) == yb)
            samples_seen += batch_size

        train_loss = epoch_loss / samples_seen
        train_acc = epoch_correct / samples_seen
        test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=args.batch_size)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    model.save(args.save_path)
    print(f"Saved weights to {args.save_path}")

    y_test_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_test_pred, num_classes=10)
    print("Confusion Matrix:")
    print(cm)

    if not args.no_visuals:
        plot_training_curves(history)
        show_predictions_grid(x_test, y_test, y_test_pred, num_images=25)
        plot_confusion_matrix(cm)


if __name__ == "__main__":
    train(parse_args())
