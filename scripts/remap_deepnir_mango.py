"""
Extract mango-only annotations from the deepNIR 11-fruit YOLO dataset and
remap to our taxonomy as cls_id 7.

deepNIR (11 classes: apple, avocado, blueberry, capsicum, cherry, kiwi,
mango, orange, rockmelon, strawberry, wheat) — only mango is missing from
our taxonomy. Other deepNIR classes overlap (apple/cherry/orange/strawberry)
but our existing + henningheyen sources already cover them with more variety,
so we ignore them here to avoid extra imbalance.

Usage:
    python -m scripts.remap_deepnir_mango \\
        --src datasets/deepNIR/deepnir_11fruits \\
        --dst datasets/deepNIR_mango \\
        --mango-cls 6
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

OUR_MANGO_CLS = 7  # our taxonomy index for mango
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def remap_split(images_src: Path, labels_src: Path, images_dst: Path,
                labels_dst: Path, mango_cls: int) -> dict:
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)
    if not labels_src.exists():
        return {"missing": True}

    n_label_files = 0
    n_kept = 0
    n_mango_boxes = 0

    for label_path in labels_src.glob("*.txt"):
        n_label_files += 1
        kept_lines: list[str] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            toks = line.split()
            if len(toks) < 5:
                continue
            try:
                cls = int(toks[0])
            except ValueError:
                continue
            if cls != mango_cls:
                continue
            kept_lines.append(f"{OUR_MANGO_CLS} " + " ".join(toks[1:]))
        if not kept_lines:
            continue

        image_path = None
        for ext in IMG_EXTS:
            cand = images_src / f"{label_path.stem}{ext}"
            if cand.exists():
                image_path = cand
                break
        if image_path is None:
            continue

        (labels_dst / label_path.name).write_text(
            "\n".join(kept_lines) + "\n", encoding="utf-8"
        )
        out_image = images_dst / image_path.name
        if not out_image.exists():
            try:
                out_image.symlink_to(image_path.resolve())
            except OSError:
                shutil.copy2(image_path, out_image)
        n_kept += 1
        n_mango_boxes += len(kept_lines)

    return {
        "n_label_files": n_label_files,
        "n_kept": n_kept,
        "n_mango_boxes": n_mango_boxes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True,
                    help="Root containing per-split images/labels (auto-detected)")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--mango-cls", type=int, required=True,
                    help="Class id used for mango in the source dataset (read from data.yaml)")
    args = ap.parse_args()

    # Auto-detect split structure. Accept both "val" and "valid". Output split
    # name is normalized to "val" so it lines up with the merge script.
    split_aliases = (("train", "train"), ("val", "val"), ("valid", "val"), ("test", "test"))
    candidates_a = [(args.src / sa / "images", args.src / sa / "labels", out_name)
                    for sa, out_name in split_aliases]
    candidates_b = [(args.src / "images" / sa, args.src / "labels" / sa, out_name)
                    for sa, out_name in split_aliases]
    layouts = candidates_a if any(p[0].exists() for p in candidates_a) else candidates_b

    results = []
    for images_src, labels_src, split in layouts:
        result = remap_split(
            images_src=images_src,
            labels_src=labels_src,
            images_dst=args.dst / "images" / split,
            labels_dst=args.dst / "labels" / split,
            mango_cls=args.mango_cls,
        )
        result["split"] = split
        results.append(result)

    print("=" * 60)
    print(f"deepNIR mango-only → {args.dst}")
    print("=" * 60)
    grand_files, grand_boxes = 0, 0
    for r in results:
        if r.get("missing"):
            print(f"  [{r['split']}] (missing)")
            continue
        print(f"  [{r['split']}] kept={r['n_kept']:5d} files  mango_boxes={r['n_mango_boxes']}")
        grand_files += r["n_kept"]
        grand_boxes += r["n_mango_boxes"]
    print(f"\nTotal: {grand_files} files, {grand_boxes} mango boxes")


if __name__ == "__main__":
    main()
