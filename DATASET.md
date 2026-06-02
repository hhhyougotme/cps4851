# Dataset download (not included in the code zip)

The **~13 GB** image set is **not** in `cps4981-project-code.zip`. Each teammate downloads it once from Kaggle, then runs `ingest_dataset.py` to build `dataset/train` and `dataset/val`.

## Source (Kaggle)

| Item | Value |
|------|--------|
| **Page** | https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset |
| **Slug** | `amerzishminha/forest-fire-smoke-and-non-fire-image-dataset` |
| **Size** | ~7 GB download (zip); ~13 GB on disk after layout |
| **Classes** | Mapped to `normal` / `smoke` / `fire` by `scripts/ingest_dataset.py` |

1. Open the link and **log in** to Kaggle.  
2. Click **Download** (or accept rules first if prompted).  
3. Unzip locally, then run ingest (see below).

---

## Option A — Browser download (simplest)

1. Download the dataset zip from the Kaggle page.  
2. Unzip to e.g. `D:\data\forest-fire-smoke\` (folder should contain `train`/`test` or `img_data`, etc.).  
3. From **project root**:

```text
python scripts\ingest_dataset.py --source "D:\data\forest-fire-smoke"
```

4. Check `dataset\train\normal`, `smoke`, `fire` and `dataset\val\...` were created.  
5. Train: `python scripts\train_fire_smoke_keras.py --epochs 8`

---

## Option B — Kaggle API (CLI)

1. Kaggle → **Settings** → **API** → **Create New Token** (saves `kaggle.json`).  
2. Place `kaggle.json` in:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`
3. Install CLI and download:

```text
pip install kaggle
kaggle datasets download -d amerzishminha/forest-fire-smoke-and-non-fire-image-dataset -p ./kaggle_download --unzip
python scripts\ingest_dataset.py --source "./kaggle_download"
```

(Adjust `--source` if unzip created a nested folder; point at the directory that contains `train` or `img_data`.)

---

## Option C — Python helper script (same as Colab)

Uses [kagglehub](https://github.com/Kaggle/kagglehub). Needs API credentials:

- **New token:** env var `KAGGLE_API_TOKEN` = your `KGAT_...` token, or  
- **Legacy:** `kaggle.json` in `~/.kaggle/` as in Option B.

```text
pip install kagglehub
python scripts\download_kaggle_dataset.py
```

This downloads via Kaggle, runs ingest, and writes `dataset/train` and `dataset/val`.

---

## Option D — Google Colab (no local 7 GB download on your PC)

1. Open `colab/train_fire_smoke.ipynb` in Colab.  
2. Set **GPU** runtime; add Kaggle token in **Secrets** (`KAGGLE_API_TOKEN` or username/key).  
3. Run section **3A** to download and build `/content/dataset`.  
4. Train in the notebook, download `artifacts.zip` (`fire_smoke.keras`).  
5. On your machine: `python scripts/export_keras_to_tfjs.py ...` (see `scripts/README.txt`).

You still need images on the machine that **trains locally**; Colab only helps if training stays in the cloud.

---

## After ingest — expected layout

```text
dataset/
  train/
    normal/   smoke/   fire/
  val/
    normal/   smoke/   fire/
```

Class order must stay **normal → smoke → fire** (indices 0, 1, 2), matching the web app.

---

## Sharing with a teammate

Send them:

1. **`cps4981-project-code.zip`** (code + trained models in `models/`, no images).  
2. **This file** or the Kaggle link:  
   https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset  
3. One line: after unzip, run  
   `python scripts\ingest_dataset.py --source "<unzip folder>"`  
   or `python scripts\download_kaggle_dataset.py` if they use the API.

They do **not** need your 13 GB copy unless you choose to share it separately (USB, cloud drive, school file server).
