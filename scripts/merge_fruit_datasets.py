"""
Merge multiple fruit-detection datasets (already in our 0..9 taxonomy) into a
single YOLO-format dataset for retraining the FruitDetector.

Sources (each must already use our cls_id 0..9):
  --existing    /home/duck/PycharmProjects/yolov8_fruit_detection/data
                (apple/banana/orange/strawberry/grape/pineapple — covers 6 classes)
  --hh          datasets/henningheyen_fruit10
                (covers 9/10; mango absent; banana wildly oversampled)
  --deepnir     datasets/deepNIR_mango   (mango-only, optional)

Banana cap strategy:
  henningheyen has ~34k banana boxes in *banana-only* files (no other class)
  vs ~8k in mixed files. Dropping banana-only files keeps variety from mixed
  scenes and limits banana growth to ~11k total. Toggle with --keep-banana-only.

Output layout:
  <dst>/{train,val,test}/{images,labels}/...
  <dst>/dataset.yaml
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

OUR_CLASSES = (
    "apple", "banana", "orange", "strawberry", "grape",
    "pineapple", "watermelon", "mango", "peach", "cherry",
)
BANANA_CLS = 1

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_label_file(path: Path) -> tuple[list[str], Counter]:
    """Return (kept_lines, class_counter) for a YOLO label file."""
    counter: Counter = Counter()
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        toks = line.split()
        if len(toks) < 5:
            continue
        try:
            cls = int(toks[0])
        except ValueError:
            continue
        if not (0 <= cls < len(OUR_CLASSES)):
            continue
        counter[cls] += 1
        kept.append(line)
    return kept, counter


def find_image(stem: str, images_dir: Path) -> Path | None:
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def merge_source(
    name: str,
    images_dir: Path,
    labels_dir: Path,
    dst_root: Path,
    split: str,
    drop_banana_only: bool,
    name_prefix: str,
) -> dict:
    out_images = dst_root / split / "images"
    out_labels = dst_root / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists():
        return {"missing": True}

    counter: Counter = Counter()
    n_kept = 0
    n_dropped_banana_only = 0
    n_missing_image = 0

    for label_path in labels_dir.glob("*.txt"):
        kept_lines, file_counter = parse_label_file(label_path)
        if not kept_lines:
            continue

        if drop_banana_only and len(file_counter) == 1 and BANANA_CLS in file_counter:
            n_dropped_banana_only += 1
            continue

        image_path = find_image(label_path.stem, images_dir)
        if image_path is None:
            n_missing_image += 1
            continue

        # Prefix to avoid collisions across sources
        new_stem = f"{name_prefix}_{label_path.stem}"
        out_label = out_labels / f"{new_stem}.txt"
        out_image = out_images / f"{new_stem}{image_path.suffix.lower()}"
        out_label.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        link_or_copy(image_path, out_image)
        for cls, n in file_counter.items():
            counter[cls] += n
        n_kept += 1

    return {
        "source": name,
        "split": split,
        "n_kept": n_kept,
        "n_dropped_banana_only": n_dropped_banana_only,
        "n_missing_image": n_missing_image,
        "per_class": dict(counter),
    }


def write_dataset_yaml(dst_root: Path) -> Path:
    yaml_path = dst_root / "dataset.yaml"
    lines = [
        f"path: {dst_root.resolve()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(OUR_CLASSES)}",
        "names:",
    ]
    for i, name in enumerate(OUR_CLASSES):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--existing", type=Path,
                    default=Path("/home/duck/PycharmProjects/yolov8_fruit_detection/data"),
                    help="Original fruit_detection dataset root (with train/val/test/images and labels)")
    ap.add_argument("--hh", type=Path,
                    default=Path("datasets/henningheyen_fruit10"),
                    help="Remapped henningheyen root (with images/<split> and labels/<split>)")
    ap.add_argument("--deepnir", type=Path, default=None,
                    help="Optional remapped deepNIR mango root (same layout as hh)")
    ap.add_argument("--dst", type=Path, default=Path("datasets/fruit10_merged"))
    ap.add_argument("--keep-banana-only",
                    action="store_true",
                    help="Do NOT drop henningheyen banana-only files (default: drop them to cap banana)")
    args = ap.parse_args()

    summaries: list[dict] = []

    # Source 1: existing fruit_detection (train/val/test)/(images,labels)
    for split in ("train", "val", "test"):
        summaries.append(merge_source(
            name="existing",
            images_dir=args.existing / split / "images",
            labels_dir=args.existing / split / "labels",
            dst_root=args.dst,
            split=split,
            drop_banana_only=False,  # keep everything from existing
            name_prefix="ex",
        ))

    # Source 2: henningheyen — layout is images/<split> and labels/<split>
    for split in ("train", "val", "test"):
        summaries.append(merge_source(
            name="henningheyen",
            images_dir=args.hh / "images" / split,
            labels_dir=args.hh / "labels" / split,
            dst_root=args.dst,
            split=split,
            drop_banana_only=(split in ("train", "val") and not args.keep_banana_only),
            name_prefix="hh",
        ))

    # Source 3: deepNIR (mango)
    if args.deepnir is not None:
        for split in ("train", "val", "test"):
            summaries.append(merge_source(
                name="deepnir",
                images_dir=args.deepnir / "images" / split,
                labels_dir=args.deepnir / "labels" / split,
                dst_root=args.dst,
                split=split,
                drop_banana_only=False,
                name_prefix="dn",
            ))

    write_dataset_yaml(args.dst)

    print("=" * 70)
    print(f"Merged → {args.dst}")
    print("=" * 70)
    grand_split: dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
    for s in summaries:
        if s.get("missing"):
            continue
        print(f"[{s['source']:12s} | {s['split']:5s}] kept={s['n_kept']:5d}  "
              f"dropped_banana_only={s['n_dropped_banana_only']:4d}  missing_img={s['n_missing_image']}")
        for cls, n in s["per_class"].items():
            grand_split[s["split"]][cls] += n

    print("\n=== Per-split, per-class totals ===")
    print(f"  {'cls':>3}  {'name':12s}  {'train':>8s}  {'val':>6s}  {'test':>6s}  {'total':>8s}")
    for cls in range(len(OUR_CLASSES)):
        tr = grand_split["train"].get(cls, 0)
        va = grand_split["val"].get(cls, 0)
        te = grand_split["test"].get(cls, 0)
        print(f"  {cls:>3}  {OUR_CLASSES[cls]:12s}  {tr:>8d}  {va:>6d}  {te:>6d}  {tr+va+te:>8d}")


if __name__ == "__main__":
    main()
