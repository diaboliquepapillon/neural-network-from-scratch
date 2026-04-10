"""Flask website for drawing digits and running NumPy inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from model import NeuralNetwork


WEIGHTS_PATH = Path("weights.npz")

app = Flask(__name__)
model: NeuralNetwork | None = None
model_error: str | None = None


def load_model_once() -> None:
    """Load the trained model weights if available."""
    global model, model_error
    if model is not None or model_error is not None:
        return
    if not WEIGHTS_PATH.exists():
        model_error = (
            "weights.npz not found. Train first with: "
            "python train.py --epochs 10 --no-visuals"
        )
        return
    try:
        nn = NeuralNetwork()
        nn.load(WEIGHTS_PATH.as_posix())
        model = nn
    except Exception as exc:  # pragma: no cover
        model_error = f"Could not load weights: {exc}"


@app.route("/")
def index():
    """Render homepage."""
    load_model_once()
    return render_template("index.html", model_error=model_error)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict digit from a 784-length flattened array."""
    load_model_once()
    if model is None:
        return jsonify({"ok": False, "error": model_error or "Model unavailable."}), 400

    payload = request.get_json(silent=True) or {}
    pixels = payload.get("pixels")
    if pixels is None:
        return jsonify({"ok": False, "error": "Missing 'pixels' in request body."}), 400

    x = np.asarray(pixels, dtype=np.float32)
    if x.ndim != 1 or x.shape[0] != 784:
        return jsonify({"ok": False, "error": "Expected 784 values."}), 400

    # Shape is (1, 784) because model expects a batch dimension.
    x = x.reshape(1, 784)
    probs, _ = model.forward(x)
    probs = probs[0]
    pred = int(np.argmax(probs))

    return jsonify(
        {
            "ok": True,
            "predicted_digit": pred,
            "top3": [
                {"digit": int(i), "prob": float(probs[i])}
                for i in np.argsort(-probs)[:3]
            ],
            "probabilities": [float(p) for p in probs],
        }
    )


if __name__ == "__main__":
    # Access at http://127.0.0.1:5000
    app.run(debug=True)
