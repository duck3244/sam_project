"""
Remap the henningheyen LVIS-fruit-vegetable dataset (63 classes, LVIS ids)
to our 10-class fruit detector taxonomy.

Henningheyen labels use the LVIS class id (e.g. "11" = apple) as the YOLO
class id. We need to:

  1. Filter labels: drop annotations whose LVIS id is not one of our targets.
  2. Remap: rewrite the class id to our fruit_detection cls_id (0..9).
  3. Drop images that end up with zero remaining boxes (optional).

Usage:
    python -m scripts.remap_henningheyen \\
        --src datasets/LVIS_Fruits_And_Vegetables \\
        --dst datasets/henningheyen_fruit10 \\
        --keep-empty
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

# Our taxonomy (matches utils/detector.py FRUIT_CLASSES) — index = our cls_id
OUR_CLASSES = (
    "apple", "banana", "orange", "strawberry", "grape",
    "pineapple", "watermelon", "mango", "peach", "cherry",
)

# henningheyen sequential YAML index (0..62) → our cls_id. Duplicate
# "Strawberry" (30) / "strawberry" (57) both map to our strawberry. Mango is
# absent in the source taxonomy, so it stays uncovered here.
LVIS_ID_TO_OUR_CLS = {
    1: 0,    # apple
    6: 1,    # banana
    44: 2,   # orange/orange fruit
    30: 3,   # Strawberry (duplicate)
    57: 3,   # strawberry
    32: 4,   # grape
    51: 5,   # pineapple
    61: 6,   # watermelon
    47: 8,   # peach
    18: 9,   # cherry
    # 7 (mango) intentionally absent — no class for mango in source taxonomy
}


def remap_split(images_src: Path, labels_src: Path, images_dst: Path,
                labels_dst: Path, keep_empty: bool) -> dict:
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    if not labels_src.exists():
        return {"missing": True}

    per_class = Counter()
    n_label_files = 0
    n_kept_files = 0
    n_dropped_empty = 0

    for label_path in labels_src.glob("*.txt"):
        n_label_files += 1
        kept_lines: list[str] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if len(tokens) < 5:
                continue
            try:
                lvis_id = int(tokens[0])
            except ValueError:
                continue
            our_cls = LVIS_ID_TO_OUR_CLS.get(lvis_id)
            if our_cls is None:
                continue
            kept_lines.append(f"{our_cls} " + " ".join(tokens[1:]))
            per_class[our_cls] += 1

        if not kept_lines and not keep_empty:
            n_dropped_empty += 1
            continue

        # Locate the image that matches this label stem
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = images_src / f"{label_path.stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            n_dropped_empty += 1
            continue

        (labels_dst / label_path.name).write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""),
            encoding="utf-8",
        )
        # Symlink to save disk; fall back to copy
        target = images_dst / image_path.name
        if not target.exists():
            try:
                target.symlink_to(image_path.resolve())
            except OSError:
                shutil.copy2(image_path, target)
        n_kept_files += 1

    return {
        "n_label_files": n_label_files,
        "n_kept_files": n_kept_files,
        "n_dropped_empty": n_dropped_empty,
        "per_class": dict(per_class),
    }


def write_dataset_yaml(dst_root: Path) -> Path:
    yaml_path = dst_root / "dataset.yaml"
    lines = [
        f"path: {dst_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(OUR_CLASSES)}",
        "names:",
    ]
    for i, name in enumerate(OUR_CLASSES):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True,
                    help="Root of the henningheyen dataset (contains train/val[/test] subdirs)")
    ap.add_argument("--dst", type=Path, required=True,
                    help="Output dataset root")
    ap.add_argument("--keep-empty", action="store_true",
                    help="Keep label files whose remapped content is empty")
    args = ap.parse_args()

    images_root = args.src / "images"
    labels_root = args.src / "labels"
    splits = sorted(d.name for d in labels_root.iterdir() if d.is_dir())
    summary = {}
    for split in splits:
        # Handle the henningheyen quirk where train/ and val/ have an extra
        # nested split directory (e.g. labels/train/train/*.txt).
        labels_src = labels_root / split
        if (labels_src / split).is_dir():
            labels_src = labels_src / split
        images_src = images_root / split
        if (images_src / split).is_dir():
            images_src = images_src / split

        result = remap_split(
            images_src=images_src,
            labels_src=labels_src,
            images_dst=args.dst / "images" / split,
            labels_dst=args.dst / "labels" / split,
            keep_empty=args.keep_empty,
        )
        summary[split] = result

    yaml_path = write_dataset_yaml(args.dst)

    print("=" * 60)
    print(f"Remapped → {args.dst}")
    print(f"dataset.yaml: {yaml_path}")
    print("=" * 60)
    grand = Counter()
    for split, data in summary.items():
        if data.get("missing"):
            continue
        print(f"\n[{split}] kept {data['n_kept_files']}/{data['n_label_files']} files "
              f"(dropped {data['n_dropped_empty']} empty)")
        for cls, n in sorted(data["per_class"].items()):
            print(f"  {cls} {OUR_CLASSES[cls]:12s}: {n}")
            grand[cls] += n
    print("\n=== Grand total per class ===")
    for cls in range(len(OUR_CLASSES)):
        print(f"  {cls} {OUR_CLASSES[cls]:12s}: {grand.get(cls, 0)}")


if __name__ == "__main__":
    main()
