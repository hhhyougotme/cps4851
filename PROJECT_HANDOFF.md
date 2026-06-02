# CPS4981 / Edge AI — project handoff

**Repo root (Windows):** `e:\cps4981\project`  
**Topic:** Early wildfire/smoke screening (demo only — not a real fire alarm).  
**Deployment:** Browser **TensorFlow.js** (laptop/phone); optional **TFLite** (Raspberry Pi, etc.).

---

## 1. What this project does

- **3 classes:** `normal` / `smoke` / `fire`.  
- **UI label order** (matches model indices): `0 → Normal`, `1 → Possible smoke`, `2 → Possible fire`.  
- **Preprocessing (must match train + web):** float pixels then **`x = x / 127.0 - 1.0`**.  
- **Model:** **MobileNetV2** (ImageNet, frozen backbone) + **`Dense(3, softmax)`**.

---

## 2. Repo layout

| Path | Role |
|------|------|
| `dataset/train/{normal,smoke,fire}`, `dataset/val/...` | Train/val images |
| `scripts/ingest_dataset.py` | Raw zip/folders → layout above |
| `scripts/train_fire_smoke_keras.py` | Local train + `models/fire_smoke.keras` + TF.js export attempt |
| `scripts/export_keras_to_tfjs.py` | Keras → `tfjs-web-app/public/model/` (Colab artifacts) |
| `scripts/export_tflite_edge.py` | Keras → TFLite quantization |
| `scripts/evaluate_results.py` | Val accuracy + latency for **all** model artifacts |
| `scripts/README.txt` | Commands and workflow |
| `RESULTS.md` | Report-ready results summary |
| `results/evaluation_report.json` | Full metrics (auto-generated) |
| `DATASET.md` | Kaggle download link for teammates |
| `requirements-train.txt` | `tensorflow>=2.16,<2.20`, optional tensorflowjs notes |
| `colab/train_fire_smoke.ipynb` | Colab: TF only, no tensorflowjs |
| `tfjs-web-app/` | React + TF.js fork |
| `tfjs-web-app/public/model/` | `model.json` + weight shards |
| `ENV-NOTES.txt` | Local env notes |

---

## 3. Frontend

- Fork of IBM **tfjs-web-app** sample.  
- **`Classify.js`:** `REACT_APP_MODEL_URL`, `REACT_APP_NUM_CLASSES=3`, `REACT_APP_MODEL_FORMAT=graph`, `processImage` = `div(127).sub(1)`, live monitor tab.  
- **`package.json`:** `NODE_OPTIONS=--openssl-legacy-provider` for newer Node; `start-dev` runs `server.js` + CRA.

**Run locally** (in `tfjs-web-app`):

- `.env`: `REACT_APP_MODEL_URL=/model/model.json`, `REACT_APP_NUM_CLASSES=3`, `REACT_APP_MODEL_FORMAT=graph`  
- `npm run start-dev` → http://localhost:3000  
- Clear **IndexedDB** after model swaps.

---

## 4. Measured results (val 8210 images, CPU)

See **`RESULTS.md`** and **`results/evaluation_report.md`**. Summary:

| Artifact | Top-1 | Latency (mean) |
|----------|-------|----------------|
| `fire_smoke.keras` | 97.70% | 65.5 ms |
| `fire_smoke_float32.tflite` | 97.70% | 5.5 ms |
| `fire_smoke_dynamic.tflite` | 97.78% | 23.2 ms |
| `fire_smoke_int8.tflite` | 97.50% | 2.6 ms |

Regenerate: `python scripts/evaluate_results.py --full-val`

---

## 5. Train and export pipelines

### A) All local

```text
python scripts/train_fire_smoke_keras.py --epochs 8
```

Output: `models/fire_smoke.keras` + `tfjs-web-app/public/model/`.

### B) Colab train → local TF.js

1. Open `colab/train_fire_smoke.ipynb` with **GPU**.  
2. Data via Kaggle API or Drive `dataset.zip` (no direct access to your PC disk).  
3. Do **not** `pip install tensorflowjs` in Colab.  
4. Download **`artifacts.zip`** (`fire_smoke.keras`).  
5. On project root:

```text
python scripts/export_keras_to_tfjs.py path\to\fire_smoke.keras tfjs-web-app\public\model
```

Colab `.keras` needs **TF ≥ 2.16** on the export machine.

---

## 6. Data

- **Not in the code zip** (~13 GB). Teammates download from Kaggle — see **`DATASET.md`**.  
- **Link:** https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset  
- After download: `python scripts/ingest_dataset.py --source "<unzip>"` or `python scripts/download_kaggle_dataset.py`.  
- Maps `non fire`→normal, `Smoke`→smoke, `fire`→fire via `ingest_dataset.py`.

---

## 7. Colab + tensorflowjs pitfalls (avoided)

Installing **tensorflowjs** in Colab hit Python 3.12 / `pkg_resources` / packaging conflicts.  
**Policy:** Colab trains `.keras` only; TF.js export on Windows/conda via `export_keras_to_tfjs.py`.

---

## 8. Example prompts for a new chat

- "Follow `scripts/README.txt` to check dataset and train"  
- "Update Colab notebook Kaggle cell"  
- "Change hazard threshold in `Classify.js`"  
- "Debug `npm run start-dev` / `.env` / missing `public/model`"

---

## 9. One-line summary

3-class (normal/smoke/fire): `dataset/` + `train_fire_smoke_keras.py` or Colab notebook → `fire_smoke.keras` → `export_keras_to_tfjs.py` → `tfjs-web-app/public/model`; web env `REACT_APP_MODEL_URL=/model/model.json`, `REACT_APP_NUM_CLASSES=3`, preprocessing `x/127-1`.

---

*Handoff notes; scripts and code in the repo are authoritative.*
