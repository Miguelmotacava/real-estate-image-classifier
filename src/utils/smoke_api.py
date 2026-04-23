"""Sanity-check helper: load the best checkpoint and run a single prediction."""
from __future__ import annotations

from pathlib import Path

from api.inference import discover_best_checkpoint, load_model, predict_image


def main() -> None:
    ckpt = discover_best_checkpoint()
    print(f"Best checkpoint: {ckpt}")
    loaded = load_model(checkpoint_path=ckpt)
    sample = next(Path("dataset/validation").rglob("*.jpg"))
    payload = predict_image(loaded, sample.read_bytes())
    print({"sample": str(sample), **payload})


if __name__ == "__main__":
    main()
