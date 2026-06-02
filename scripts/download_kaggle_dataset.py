"""
Download the forest fire / smoke Kaggle dataset and build dataset/train + dataset/val.

Requires Kaggle API access (one of):
  - Environment variable KAGGLE_API_TOKEN (KGAT_... token from Kaggle Settings -> API)
  - Legacy file: %USERPROFILE%\\.kaggle\\kaggle.json  (Windows) or ~/.kaggle/kaggle.json

Dataset page:
  https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset

Usage (from project root):
  pip install kagglehub
  python scripts/download_kaggle_dataset.py
  python scripts/train_fire_smoke_keras.py --epochs 8
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "amerzishminha/forest-fire-smoke-and-non-fire-image-dataset"
KAGGLE_PAGE = (
    "https://www.kaggle.com/datasets/amerzishminha/"
    "forest-fire-smoke-and-non-fire-image-dataset"
)


def _ensure_kaggle_credentials() -> None:
    if os.environ.get("KAGGLE_API_TOKEN", "").strip():
        return
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.is_file():
        return
    raise SystemExit(
        "Kaggle credentials missing.\n"
        f"1) Open {KAGGLE_PAGE}\n"
        "2) Kaggle -> Settings -> API -> Create New Token\n"
        "3) Set env KAGGLE_API_TOKEN=KGAT_...  OR  save kaggle.json to ~/.kaggle/\n"
        "Then re-run this script."
    )


def main() -> None:
    _ensure_kaggle_credentials()
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("Install kagglehub first:  pip install kagglehub")

    print(f"Downloading: {DATASET_SLUG}")
    print(f"Page: {KAGGLE_PAGE}")
    print("(Large download ~7 GB; may take 15–40 minutes.)\n")

    path = kagglehub.dataset_download(DATASET_SLUG)
    source = Path(path)
    print(f"Kaggle cache path: {source.resolve()}\n")

    root = Path(__file__).resolve().parent.parent
    ingest = root / "scripts" / "ingest_dataset.py"
    cmd = [sys.executable, str(ingest), "--source", str(source)]
    print("Running ingest:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("\nDone. dataset/train and dataset/val are ready.")
    print("Next: python scripts/train_fire_smoke_keras.py --epochs 8")


if __name__ == "__main__":
    main()
