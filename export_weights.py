"""Export trained NumPy weights to JSON for browser inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main(npz_path: str = "weights.npz", out_path: str = "site/weights.json") -> None:
    src = Path(npz_path)
    dst = Path(out_path)
    if not src.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Train first with `python train.py --epochs 10 --no-visuals`."
        )

    params = np.load(src)
    payload = {
        "W1": params["W1"].tolist(),
        "b1": params["b1"].tolist(),
        "W2": params["W2"].tolist(),
        "b2": params["b2"].tolist(),
        "W3": params["W3"].tolist(),
        "b3": params["b3"].tolist(),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Exported browser weights to {dst}")


if __name__ == "__main__":
    main()
