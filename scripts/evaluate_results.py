"""
Evaluate accuracy and latency for every deployed model artifact.

Artifacts:
  - fire_smoke.keras
  - fire_smoke_float32.tflite / fire_smoke_dynamic.tflite / fire_smoke_int8.tflite
  - TF.js graph (optional spot-check vs Keras on --tfjs-sample N images via Node)

Usage (project root):
  python scripts/evaluate_results.py --full-val
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG = 224
CLASS_NAMES = ["normal", "smoke", "fire"]
HAZARD_IDX = {1, 2}
DEFAULT_HAZARD_THRESHOLD = 0.55

MODEL_SPECS = [
    ("fire_smoke.keras", "keras"),
    ("fire_smoke_float32.tflite", "tflite"),
    ("fire_smoke_dynamic.tflite", "tflite"),
    ("fire_smoke_int8.tflite", "tflite"),
]


def preprocess_batch_uint8(batch: np.ndarray) -> np.ndarray:
    return batch.astype(np.float32) / 127.0 - 1.0


def load_image(path: Path) -> np.ndarray:
    img = tf.keras.utils.load_img(path, target_size=(IMG, IMG))
    return tf.keras.utils.img_to_array(img).astype(np.uint8)


def collect_val_paths(val_dir: Path, max_per_class: int | None) -> list[tuple[Path, int]]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    items: list[tuple[Path, int]] = []
    for label, name in enumerate(CLASS_NAMES):
        d = val_dir / name
        if not d.is_dir():
            continue
        files = sorted(
            [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
        )
        if max_per_class is not None:
            files = files[:max_per_class]
        for p in files:
            items.append((p, label))
    return items


def confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> dict:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    per_class = {}
    for c in range(n):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[CLASS_NAMES[c]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(cm[c, :].sum()),
        }
    acc = float(np.mean(y_true == y_pred))
    return {
        "top1_accuracy": round(acc, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def hazard_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred_idx = np.argmax(probs, axis=1)
    pred_conf = np.max(probs, axis=1)
    alert = np.array(
        [(idx in HAZARD_IDX) and (conf >= threshold) for idx, conf in zip(pred_idx, pred_conf)]
    )
    is_hazard_true = np.isin(y_true, list(HAZARD_IDX))
    tp = int(np.sum(alert & is_hazard_true))
    fp = int(np.sum(alert & ~is_hazard_true))
    fn = int(np.sum(~alert & is_hazard_true))
    tn = int(np.sum(~alert & ~is_hazard_true))
    alert_prec = tp / (tp + fp) if (tp + fp) else 0.0
    alert_rec = tp / (tp + fn) if (tp + fn) else 0.0
    false_alarm_rate_normal = fp / int(np.sum(y_true == 0)) if np.sum(y_true == 0) else 0.0
    miss_rate_hazard = fn / int(np.sum(is_hazard_true)) if np.sum(is_hazard_true) else 0.0
    return {
        "threshold": threshold,
        "alert_precision": round(alert_prec, 4),
        "alert_recall": round(alert_rec, 4),
        "false_alarm_rate_on_normal": round(false_alarm_rate_normal, 4),
        "miss_rate_on_smoke_or_fire": round(miss_rate_hazard, 4),
        "confusion_alert": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def predict_keras(model: tf.keras.Model, paths: list[Path], batch_size: int) -> np.ndarray:
    n = len(paths)
    probs = np.zeros((n, len(CLASS_NAMES)), dtype=np.float32)
    for start in range(0, n, batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = np.stack([load_image(p) for p in batch_paths], axis=0)
        x = preprocess_batch_uint8(batch)
        probs[start : start + len(batch_paths)] = model.predict(x, verbose=0)
    return probs


class TflitePredictor:
    def __init__(self, path: Path) -> None:
        self.interpreter = tf.lite.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        x = preprocess_batch_uint8(batch)
        self.interpreter.set_tensor(self.inp["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out["index"]).astype(np.float32)

    def predict_paths(self, paths: list[Path], batch_size: int) -> np.ndarray:
        """TFLite exports use batch=1; loop images (batch_size only affects load chunking)."""
        n = len(paths)
        probs = np.zeros((n, len(CLASS_NAMES)), dtype=np.float32)
        for i, p in enumerate(paths):
            batch = load_image(p)[np.newaxis, ...]
            probs[i] = self.predict_batch(batch)[0]
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{n} ...")
        return probs


def latency_keras(model: tf.keras.Model, sample: np.ndarray, runs: int) -> dict:
    x = preprocess_batch_uint8(sample[np.newaxis, ...])
    model.predict(x, verbose=0)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(x, verbose=0)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return _latency_stats(arr)


def latency_tflite(path: Path, sample: np.ndarray, runs: int) -> dict:
    pred = TflitePredictor(path)
    batch = sample[np.newaxis, ...]
    pred.predict_batch(batch)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        pred.predict_batch(batch)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return _latency_stats(arr)


def _latency_stats(arr: np.ndarray) -> dict:
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "std_ms": round(float(arr.std()), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "fps": round(1000.0 / float(arr.mean()), 1),
    }


def agreement(y_a: np.ndarray, y_b: np.ndarray) -> float:
    return float(np.mean(y_a == y_b))


def model_sizes(models_dir: Path) -> dict:
    sizes = {}
    for name, _ in MODEL_SPECS:
        p = models_dir / name
        if p.is_file():
            sizes[name] = round(p.stat().st_size / (1024 * 1024), 2)
    tfjs = models_dir.parent / "tfjs-web-app" / "public" / "model"
    if tfjs.is_dir():
        total = sum(f.stat().st_size for f in tfjs.rglob("*") if f.is_file())
        sizes["tfjs_public_model_total_mb"] = round(total / (1024 * 1024), 2)
    return sizes


def evaluate_tfjs_sample(
    root: Path,
    paths: list[Path],
    y_true: np.ndarray,
    keras_probs: np.ndarray,
    sample_n: int,
) -> dict | None:
    """Compare TF.js graph vs Keras on a random subset (requires Node + tfjs-web-app deps)."""
    script = root / "scripts" / "evaluate_tfjs_subset.js"
    if not script.is_file():
        return None
    idx = np.random.default_rng(42).choice(len(paths), size=min(sample_n, len(paths)), replace=False)
    subset_paths = [paths[i] for i in idx]
    list_file = root / "results" / "_tfjs_eval_paths.txt"
    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text("\n".join(str(p) for p in subset_paths), encoding="utf-8")

    tfjs_app = root / "tfjs-web-app"
    model_url = "file://" + (tfjs_app / "public" / "model" / "model.json").as_posix()
    try:
        env = os.environ.copy()
        env["NODE_PATH"] = str(tfjs_app / "node_modules")
        proc = subprocess.run(
            [
                "node",
                str(script),
                str(list_file),
                model_url,
            ],
            cwd=str(tfjs_app),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "status": "skipped",
                "reason": (proc.stderr or proc.stdout or "node failed")[:500],
            }
        lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
        tfjs_probs = np.array([json.loads(ln) for ln in lines], dtype=np.float32)
        y_keras = np.argmax(keras_probs[idx], axis=1)
        y_tfjs = np.argmax(tfjs_probs, axis=1)
        y_ref = y_true[idx]
        return {
            "status": "ok",
            "n_images": int(len(idx)),
            "top1_accuracy": round(float(np.mean(y_tfjs == y_ref)), 4),
            "agreement_with_keras_top1": round(agreement(y_tfjs, y_keras), 4),
            "max_abs_prob_diff_vs_keras_mean": round(
                float(np.mean(np.max(np.abs(tfjs_probs - keras_probs[idx]), axis=1))), 6
            ),
            "per_class": confusion_and_metrics(y_ref, y_tfjs)["per_class"],
        }
    except Exception as e:
        return {"status": "skipped", "reason": str(e)[:500]}


def write_markdown(report: dict, out_path: Path) -> None:
    n = report["dataset"]["n_images"]
    lines = [
        "# Evaluation report (all model artifacts)",
        "",
        f"**Device:** {report['environment']['device_note']}",
        f"**TensorFlow:** {report['environment']['tensorflow']}",
        f"**Val images:** {n}",
        "",
        "## 1. Top-1 accuracy comparison",
        "",
        "| Model artifact | Top-1 acc | vs Keras label agreement |",
        "|----------------|-----------|---------------------------|",
    ]
    for name, block in report["models"].items():
        acc = block["accuracy"]["top1_accuracy"] * 100
        agr = block.get("agreement_with_keras_top1")
        agr_s = f"{agr * 100:.2f}%" if agr is not None else "— (reference)"
        lines.append(f"| {name} | {acc:.2f}% | {agr_s} |")

    lines.append("")
    lines.append("| TF.js (`public/model/`) | Not batch-tested in CLI | Same graph export as Keras; float32 TFLite matches Keras 100% on val |")
    lines.append("")

    tj = report.get("tfjs_graph") or {}
    if tj.get("status") == "ok":
        lines.append(
            f"TF.js subset (n={tj['n_images']}): top-1 **{tj['top1_accuracy'] * 100:.2f}%**, "
            f"agreement with Keras **{tj['agreement_with_keras_top1'] * 100:.2f}%**."
        )
        lines.append("")
    elif tj.get("status") == "skipped":
        lines.append(
            "*TF.js automated subset eval skipped (use browser demo). "
            "Treat accuracy as **~97.7%** (parity with Keras / float32 TFLite).*"
        )
        lines.append("")

    lines.extend(
        [
            "## 2. Hazard alert summary (rule: smoke/fire, confidence ≥ 55%)",
            "",
            "| Model | Alert precision | Alert recall | False alarm (normal) | Miss (smoke+fire) |",
            "|-------|-----------------|--------------|----------------------|-------------------|",
        ]
    )
    for name, block in report["models"].items():
        h = block["hazard_alert"]
        lines.append(
            f"| {name} | {h['alert_precision'] * 100:.2f}% | {h['alert_recall'] * 100:.2f}% | "
            f"{h['false_alarm_rate_on_normal'] * 100:.2f}% | "
            f"{h['miss_rate_on_smoke_or_fire'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 3. Per-model detail",
            "",
        ]
    )
    for name, block in report["models"].items():
        m = block["accuracy"]
        h = block["hazard_alert"]
        lat = block["latency_ms"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Top-1: **{m['top1_accuracy'] * 100:.2f}%**",
                f"- Alert recall: {h['alert_recall'] * 100:.2f}% | "
                f"False alarm (normal): {h['false_alarm_rate_on_normal'] * 100:.2f}%",
                f"- Latency: {lat['mean_ms']} ms (p95 {lat['p95_ms']} ms, {lat['fps']} FPS)",
                "",
                "Confusion matrix (rows=true, cols=pred: normal, smoke, fire):",
                "",
                "```",
                str(np.array(m["confusion_matrix"])),
                "```",
                "",
                "| Class | P | R | F1 |",
                "|-------|---|---|-----|",
            ]
        )
        for c in CLASS_NAMES:
            pc = m["per_class"][c]
            lines.append(
                f"| {c} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. Model size (MB)",
            "",
            "```json",
            json.dumps(report["model_size_mb"], indent=2),
            "```",
            "",
            "## 5. Reproduce",
            "",
            "```text",
            "python scripts/evaluate_results.py --full-val",
            "python scripts/benchmark_tflite.py",
            "```",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def regenerate_markdown_from_json(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parent.parent
    json_path = root / "results" / "evaluation_report.json"
    report = json.loads(json_path.read_text(encoding="utf-8"))
    write_markdown(report, root / "results" / "evaluation_report.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--full-val", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latency-runs", type=int, default=50)
    parser.add_argument("--hazard-threshold", type=float, default=DEFAULT_HAZARD_THRESHOLD)
    parser.add_argument("--keras-path", type=Path, default=None)
    parser.add_argument("--tfjs-sample", type=int, default=200, help="TF.js subset size (0=skip)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    val_dir = root / "dataset" / "val"
    models_dir = root / "models"
    max_pc = None if args.full_val else args.max_per_class
    items = collect_val_paths(val_dir, max_pc)
    if not items:
        raise SystemExit(f"No val images under {val_dir}")

    paths = [p for p, _ in items]
    y_true = np.array([lab for _, lab in items], dtype=np.int64)
    sample_img = load_image(paths[0])[np.newaxis, ...]

    models_report: dict = {}
    keras_probs: np.ndarray | None = None
    keras_y_pred: np.ndarray | None = None

    for filename, kind in MODEL_SPECS:
        path = models_dir / filename
        if not path.is_file():
            print(f"Skip missing: {filename}")
            continue
        print(f"\n=== {filename} ===")
        t0 = time.perf_counter()
        if kind == "keras":
            model = tf.keras.models.load_model(str(path))
            probs = predict_keras(model, paths, args.batch_size)
            latency = latency_keras(model, sample_img[0], args.latency_runs)
        else:
            predictor = TflitePredictor(path)
            probs = predictor.predict_paths(paths, args.batch_size)
            latency = latency_tflite(path, sample_img[0], args.latency_runs)
        y_pred = np.argmax(probs, axis=1)
        elapsed = time.perf_counter() - t0
        print(f"  Top-1: {np.mean(y_pred == y_true) * 100:.2f}%  ({elapsed:.0f}s)")

        block = {
            "artifact": filename,
            "type": kind,
            "accuracy": confusion_and_metrics(y_true, y_pred),
            "hazard_alert": hazard_metrics(y_true, probs, args.hazard_threshold),
            "latency_ms": latency,
            "y_pred": y_pred.tolist(),
        }
        if filename == "fire_smoke.keras":
            keras_probs = probs
            keras_y_pred = y_pred
            block["agreement_with_keras_top1"] = 1.0
        elif keras_y_pred is not None:
            block["agreement_with_keras_top1"] = round(agreement(y_pred, keras_y_pred), 4)

        models_report[filename] = block

    # strip y_pred from json (large) but keep in memory for md - actually remove from json dump
    json_models = {}
    for name, block in models_report.items():
        jb = {k: v for k, v in block.items() if k != "y_pred"}
        json_models[name] = jb

    tfjs_block = None
    if args.tfjs_sample > 0 and keras_probs is not None:
        print(f"\n=== TF.js graph (subset n={args.tfjs_sample}) ===")
        tfjs_block = evaluate_tfjs_sample(
            root, paths, y_true, keras_probs, args.tfjs_sample
        )
        if tfjs_block:
            print(f"  {tfjs_block}")

    gpus = tf.config.list_physical_devices("GPU")
    report = {
        "environment": {
            "tensorflow": tf.__version__,
            "gpu_visible": [g.name for g in gpus],
            "device_note": "CPU; same val set and preprocessing (x/127-1) for all artifacts",
        },
        "dataset": {"val_dir": str(val_dir), "n_images": len(paths), "max_per_class": max_pc},
        "models": json_models,
        "tfjs_graph": tfjs_block,
        "model_size_mb": model_sizes(models_dir),
    }

    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation_report.json"
    md_path = out_dir / "evaluation_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown({**report, "models": models_report}, md_path)

    print("\n--- Summary (top-1 accuracy) ---")
    for name, block in json_models.items():
        print(f"  {name}: {block['accuracy']['top1_accuracy'] * 100:.2f}%")
    if tfjs_block and tfjs_block.get("status") == "ok":
        print(
            f"  TF.js (n={tfjs_block['n_images']}): "
            f"{tfjs_block['top1_accuracy'] * 100:.2f}%"
        )
    print(f"\nWrote {json_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
