"""
Simple CPU inference benchmark for models/fire_smoke_*.tflite (224x224, same preprocessing as train/web).

Usage (project root):
  python scripts/benchmark_tflite.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG = 224
CLASS_NAMES = ["normal", "smoke", "fire"]


def preprocess(batch_hwc: np.ndarray) -> np.ndarray:
    return batch_hwc.astype(np.float32) / 127.0 - 1.0


def benchmark(path: Path, runs: int = 30) -> tuple[float, float]:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    dummy = preprocess(np.random.randint(0, 256, (1, IMG, IMG, 3), dtype=np.uint8))
    # warmup
    interpreter.set_tensor(inp["index"], dummy)
    interpreter.invoke()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(inp["index"], dummy)
        interpreter.invoke()
        interpreter.get_tensor(out["index"])
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "models"
    keras_mb = (root / "fire_smoke.keras").stat().st_size / (1024 * 1024) if (root / "fire_smoke.keras").is_file() else 0
    print(f"Keras checkpoint: fire_smoke.keras  ({keras_mb:.2f} MB)")
    print(f"{'artifact':<28} {'size (KB)':>12} {'mean ms':>10} {'std ms':>8}")
    print("-" * 62)
    for name in ("fire_smoke_float32", "fire_smoke_dynamic", "fire_smoke_int8"):
        p = root / f"{name}.tflite"
        if not p.is_file():
            print(f"{name}.tflite  (missing)")
            continue
        kb = p.stat().st_size / 1024
        mean_ms, std_ms = benchmark(p)
        print(f"{name + '.tflite':<28} {kb:12.1f} {mean_ms:10.2f} {std_ms:8.2f}")


if __name__ == "__main__":
    main()
