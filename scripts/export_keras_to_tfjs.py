"""
Export a full Keras model (.keras) to TensorFlow.js Graph format (model.json + *.bin).

Keras 3 checkpoints use convert_keras_model_to_graph_model; the web app needs
tf.loadGraphModel + REACT_APP_MODEL_FORMAT=graph.

tensorflowjs imports TFDF; this script stubs that module. Install jax/jaxlib
matching NumPy (e.g. 0.4.23).

Usage (from project root):
  python scripts/export_keras_to_tfjs.py models/fire_smoke.keras tfjs-web-app/public/model
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path


def _stub_optional_tfjs_imports() -> None:
    if "tensorflow_decision_forests" not in sys.modules:
        sys.modules["tensorflow_decision_forests"] = types.ModuleType(
            "tensorflow_decision_forests"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Keras 3 -> TF.js graph model in public/model")
    parser.add_argument("keras_path", type=Path, help="e.g. models/fire_smoke.keras")
    parser.add_argument(
        "out_dir",
        type=Path,
        help="e.g. tfjs-web-app/public/model",
    )
    args = parser.parse_args()

    if not args.keras_path.is_file():
        raise SystemExit(f"Model file not found: {args.keras_path.resolve()}")

    os.environ["TF_USE_LEGACY_KERAS"] = "0"
    _stub_optional_tfjs_imports()

    import tensorflow as tf
    from tensorflowjs.converters.tf_saved_model_conversion_v2 import (
        convert_keras_model_to_graph_model,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(str(args.keras_path))
    convert_keras_model_to_graph_model(model, str(args.out_dir))
    print(f"Exported TF.js (graph): {args.out_dir.resolve()}")
    print(
        "Frontend .env: REACT_APP_MODEL_URL=/model/model.json  "
        "REACT_APP_NUM_CLASSES=3  REACT_APP_MODEL_FORMAT=graph  "
        "then restart npm run start-dev"
    )


if __name__ == "__main__":
    main()
