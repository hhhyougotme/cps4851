# CPS4851 — Wildfire / smoke edge screening (demo)

3-class image classifier (**normal** / **smoke** / **fire**) with browser **TensorFlow.js** demo and optional **TFLite** exports for edge devices (e.g. UAV companion computer).

## Quick start

### 1. Dataset

Not included in this repo (~13 GB). Download from Kaggle and ingest:

- https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset  
- See [DATASET.md](DATASET.md)

```bash
python scripts/ingest_dataset.py --source "path/to/unzipped/data"
```

### 2. Train (Python)

```bash
pip install -r requirements-train.txt
python scripts/train_fire_smoke_keras.py --epochs 8
```

Outputs `models/fire_smoke.keras` and attempts TF.js export to `tfjs-web-app/public/model/`.

If TF.js export fails:

```bash
python scripts/export_keras_to_tfjs.py models/fire_smoke.keras tfjs-web-app/public/model
```

TFLite (optional):

```bash
python scripts/export_tflite_edge.py --mode int8
```

### 3. Web demo

Create `tfjs-web-app/.env`:

```env
REACT_APP_MODEL_URL=/model/model.json
REACT_APP_NUM_CLASSES=3
REACT_APP_MODEL_FORMAT=graph
```

```bash
cd tfjs-web-app
npm install
npm run start-dev
```

Open http://localhost:3000

### 4. Evaluation / results

```bash
python scripts/evaluate_results.py --full-val
```

See [RESULTS.md](RESULTS.md) and [results/evaluation_report.md](results/evaluation_report.md).

## Repo layout

| Path | Description |
|------|-------------|
| `scripts/` | Train, export, ingest, evaluate |
| `tfjs-web-app/` | React + TF.js UI |
| `colab/` | Optional Colab training notebook |
| `results/` | Evaluation metrics (JSON + MD) |

## Disclaimer

Educational prototype only — not for real fire-alarm or aviation safety use.
