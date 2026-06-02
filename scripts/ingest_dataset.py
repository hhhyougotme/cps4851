"""
Organize a Kaggle unzip tree into dataset/train and dataset/val.

Supported layouts under --source:
  - .../img_data/train/<class> and .../img_data/test/<class>
  - .../train/<class>, .../val/<class>
  - Top-level folders normal|default|no_fire, smoke, fire, etc.

Maps folder names to: normal, smoke, fire
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# Extra aliases (spaces/underscores normalized in folder_to_class)
NAME_MAP = {
    "normal": "normal",
    "default": "normal",
    "non_fire": "normal",
    "no_fire": "normal",
    "neither": "normal",
    "not_fire": "normal",
    "smoke": "smoke",
    "fire": "fire",
}


def folder_to_class(folder_name: str) -> str | None:
    """e.g. FOREST_FIRE dataset folders 'non fire', 'Smoke', 'fire'."""
    n = folder_name.strip().lower().replace("_", " ")
    if n in ("non fire", "nonfire", "no fire", "nofire", "not fire"):
        return "normal"
    if n == "smoke":
        return "smoke"
    if n == "fire":
        return "fire"
    return NAME_MAP.get(n)


def collect_from_split_parent(split_parent: Path, pools: dict[str, list[Path]]) -> None:
    """Each child of split_parent is one class, e.g. train/default, train/fire."""
    for class_dir in split_parent.iterdir():
        if not class_dir.is_dir():
            continue
        target = folder_to_class(class_dir.name)
        if target is None:
            continue
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in EXT:
                pools[target].append(f)


def collect_all(source: Path) -> dict[str, list[Path]]:
    pools: dict[str, list[Path]] = defaultdict(list)
    source = source.resolve()

    img_data = source / "img_data"
    if img_data.is_dir():
        for split in ("train", "test", "val"):
            p = img_data / split
            if p.is_dir():
                collect_from_split_parent(p, pools)
        if any(pools.values()):
            return pools

    for split in ("train", "test", "val"):
        p = source / split
        if p.is_dir():
            collect_from_split_parent(p, pools)
    if any(pools.values()):
        return pools

    collect_from_split_parent(source, pools)
    return pools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Unzipped dataset root (contains img_data or train subdirs)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation fraction per class",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_train = root / "dataset" / "train"
    out_val = root / "dataset" / "val"

    if not args.source.is_dir():
        raise SystemExit(f"Directory not found: {args.source}")

    pools = collect_all(args.source)
    for c in ("normal", "smoke", "fire"):
        pools.setdefault(c, [])

    print("Counts (merged, before split):")
    for c in ("normal", "smoke", "fire"):
        print(f"  {c}: {len(pools[c])} images")

    if sum(len(pools[c]) for c in pools) == 0:
        raise SystemExit(
            "No images found. Check --source points at the unzip root "
            "and folder names match the mapping table."
        )

    random.seed(args.seed)
    for split in (out_train, out_val):
        for c in ("normal", "smoke", "fire"):
            d = split / c
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    for c in ("normal", "smoke", "fire"):
        files = list(pools[c])
        random.shuffle(files)
        n_val = int(len(files) * args.val_ratio)
        val_files = files[:n_val]
        train_files = files[n_val:]
        for i, f in enumerate(train_files):
            shutil.copy2(f, out_train / c / f"{c}_{i:06d}{f.suffix.lower()}")
        for i, f in enumerate(val_files):
            shutil.copy2(f, out_val / c / f"{c}_{i:06d}{f.suffix.lower()}")

    print(f"\nWrote:\n  {out_train}\n  {out_val}")
    print("Next: python scripts\\train_fire_smoke_keras.py")


if __name__ == "__main__":
    main()
