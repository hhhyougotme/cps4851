Smoke / fire / normal — 3-class training and export
===================================================

Layout (matches web UI label order):
  dataset/train/normal   -> class index 0 "Normal"
  dataset/train/smoke    -> class index 1 "Possible smoke"
  dataset/train/fire     -> class index 2 "Possible fire"
  dataset/val/           same structure

Preprocessing (same as Classify.js): after float pixels, x = x / 127.0 - 1.0

1) Create a conda env (Python 3.10 or 3.11 recommended):
   conda create -n tf-fire python=3.11 -y
   conda activate tf-fire
   pip install -r requirements-train.txt

   GPU training on NVIDIA (local):
   - Install the latest GPU driver.
   - Verify:
       python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
     If you see [], no GPU is visible: see https://www.tensorflow.org/install/pip for CUDA/cuDNN
     matching your TF version, or use WSL2 (Ubuntu) with the same conda env (common on Windows).
   - Train:
       cd /d e:\cps4981\project
       python scripts\train_fire_smoke_keras.py --epochs 8
     The script prints visible GPU devices; with a GPU, the frozen MobileNet stage is much faster than CPU.

2) Get the dataset (~13 GB on disk; NOT in the small code zip)
   See DATASET.md for the Kaggle link and all download options.
   Page: https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset

   A) Browser: download zip from Kaggle, unzip, then:
      python scripts\ingest_dataset.py --source "path\to\unzipped\root"
   B) API helper (needs Kaggle token / kaggle.json):
      pip install kagglehub
      python scripts\download_kaggle_dataset.py
   C) Manual copy into dataset/train/{normal,smoke,fire} and dataset/val/...

3) From project root e:\cps4981\project:
   python scripts\train_fire_smoke_keras.py --epochs 8

4) Writes models\fire_smoke.keras and tries TF.js export; on dependency conflicts run:
      python scripts\export_keras_to_tfjs.py models\fire_smoke.keras tfjs-web-app\public\model
   Keras 3 exports as graph-model (not legacy Layers). In tfjs-web-app\.env set:
      REACT_APP_MODEL_URL=/model/model.json
      REACT_APP_NUM_CLASSES=3
      REACT_APP_MODEL_FORMAT=graph
   Export env needs tensorflowjs (4.20+ recommended) and jax/jaxlib (e.g. jax==0.4.23 with NumPy 1.26).

   Colab (colab\train_fire_smoke.ipynb) without tensorflowjs:
   - Notebook saves fire_smoke.keras and artifacts.zip;
   - Place .keras under models\, then from project root:
       pip install -r requirements-train.txt
       python scripts\export_keras_to_tfjs.py models\fire_smoke.keras tfjs-web-app\public\model

5) Enable 3-class UI: configure tfjs-web-app\.env (include MODEL_FORMAT=graph for graph export).
   Restart npm run start-dev; clear site IndexedDB or use a private window if an old Layers model was cached.

6) Results / evaluation (all models: Keras + 3x TFLite on val set):
      python scripts\evaluate_results.py --full-val
   Outputs: results\evaluation_report.json, results\evaluation_report.md
   Summary for report: RESULTS.md

7) Disclaimer: demo prototype only — not for real fire-alarm decisions.

---

Edge AI extras (quantization / pruning / Raspberry Pi)
======================================================

Current flow: train -> TF.js web. For smaller devices (e.g. Raspberry Pi), export the same Keras model
to TensorFlow Lite and run Python + tflite_runtime; this does not conflict with the browser.

Quantization (post-training, no retrain)
  - Float32 TFLite baseline:
      python scripts\export_tflite_edge.py --mode float32
  - Dynamic range (smaller size):
      python scripts\export_tflite_edge.py --mode dynamic
  - INT8 post-training (calibration from dataset/train):
      python scripts\export_tflite_edge.py --mode int8
  Outputs: models\fire_smoke_float32.tflite, fire_smoke_dynamic.tflite, fire_smoke_int8.tflite
  Local CPU benchmark (not Pi):
      python scripts\benchmark_tflite.py
  Compare Keras size, each .tflite size, and per-frame latency in your report.

Pruning
  - Usually done during training with tensorflow_model_optimization, then export to TFLite / TF.js.
  - Good topic for "future work" in the report.

Raspberry Pi
  - pip install tflite-runtime (or full tensorflow), load .tflite with Interpreter,
    input 224x224, preprocessing x/127-1 (same as web).

TF.js (optional)
  - tensorflowjs_converter supports quantized export; see official docs to shrink browser model size.
