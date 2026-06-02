"""
Export models/fire_smoke.keras to TensorFlow Lite for edge devices (e.g. Raspberry Pi
with Python + tflite_runtime).

Modes:
  float32  — baseline, best compatibility
  dynamic  — dynamic-range quantization (smaller, no calibration set)
  int8     — full integer inference; sample images from dataset/train for calibration

Notes:
  - Browser still uses TF.js; TFLite is for ARM/Linux edge boards.
  - Pruning belongs in the training pipeline (tensorflow_model_optimization); not here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_NAMES = ["normal", "smoke", "fire"]
IMG = 224


def preprocess_array(x: np.ndarray) -> np.ndarray:
    """NHWC uint8 -> float32, /127-1, same as web app."""
    return x.astype(np.float32) / 127.0 - 1.0


def load_image_paths(train_dir: Path, max_per_class: int = 15) -> list[Path]:
    paths: list[Path] = []
    for name in CLASS_NAMES:
        d = train_dir / name
        if not d.is_dir():
            continue
        files = sorted(
            [
                p
                for p in d.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
            ]
        )[:max_per_class]
        paths.extend(files)
    return paths


def representative_generator(image_paths: list[Path]):
    for p in image_paths:
        img = tf.keras.utils.load_img(p, target_size=(IMG, IMG))
        arr = tf.keras.utils.img_to_array(img)
        arr = preprocess_array(np.expand_dims(arr, 0))
        yield [arr]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("float32", "dynamic", "int8"),
        default="dynamic",
        help="float32 | dynamic (try first) | int8 (needs train images for calibration)",
    )
    parser.add_argument(
        "--keras-path",
        type=Path,
        default=None,
        help="default models/fire_smoke.keras",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    keras_path = args.keras_path or (root / "models" / "fire_smoke.keras")
    if not keras_path.is_file():
        raise SystemExit(
            f"Model not found: {keras_path}. Run train_fire_smoke_keras.py first."
        )

    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.mode == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif args.mode == "int8":
        train_dir = root / "dataset" / "train"
        paths = load_image_paths(train_dir)
        if len(paths) < 5:
            raise SystemExit(
                "int8 mode needs sample images under dataset/train per class; "
                "or use --mode dynamic."
            )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep_dataset():
            yield from representative_generator(paths)

        converter.representative_dataset = rep_dataset
        # If conversion fails, fall back to --mode dynamic

    tflite_model = converter.convert()
    out_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.mode
    out_path = out_dir / f"fire_smoke_{suffix}.tflite"
    out_path.write_bytes(tflite_model)

    kb = len(tflite_model) / 1024
    print(f"Wrote {out_path} (~{kb:.1f} KB)")
    print(
        "On Raspberry Pi: pip install tflite-runtime (or full tensorflow), "
        "then load this .tflite with Interpreter."
    )


if __name__ == "__main__":
    main()
