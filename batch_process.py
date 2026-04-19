#!/usr/bin/env python3
"""
Auto-labeling batch processor:
    images/ → YOLO detect → SAM box prompt → YOLO-seg .txt labels.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from utils import (
    SAMPredictor,
    FruitDetector,
    FRUIT_CLASSES,
    DEFAULT_FRUIT_WEIGHTS,
    write_dataset_yaml,
)
from utils.label_pipeline import FruitLabelPipeline


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def get_image_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM + YOLO auto-labeling (YOLO-seg export)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sam_model", choices=["vit_h", "vit_l", "vit_b"], default="vit_b")
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--det_weights", type=str, default=str(DEFAULT_FRUIT_WEIGHTS))
    parser.add_argument("--det_conf", type=float, default=0.15)
    parser.add_argument("--max_size", type=int, default=1024)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--multi_contour", action="store_true",
                        help="Preserve disjoint blobs per object (>= --min_area_ratio of largest)")
    parser.add_argument("--min_area_ratio", type=float, default=0.1,
                        help="Min area ratio for secondary contours (only with --multi_contour)")
    parser.add_argument("--write_dataset_yaml", action="store_true",
                        help="Emit dataset.yaml next to the label output")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print("=" * 60)
    print("Auto-labeling pipeline (YOLO-seg)")
    print(f"  input        : {input_dir}  ({len(image_files)} files)")
    print(f"  output       : {labels_dir}")
    print(f"  SAM model    : {args.sam_model}")
    print(f"  detector     : {args.det_weights}")
    print(f"  classes      : {FRUIT_CLASSES}")
    print("=" * 60)

    sam = SAMPredictor(model_type=args.sam_model, checkpoint_path=args.sam_checkpoint)
    detector = FruitDetector(weights=args.det_weights, conf=args.det_conf)
    pipeline = FruitLabelPipeline(
        sam=sam, detector=detector,
        epsilon=args.epsilon, max_image_size=args.max_size,
        multi_contour=args.multi_contour, min_area_ratio=args.min_area_ratio,
    )

    total_labels = 0
    total_detections = 0
    failed: list[tuple[str, str]] = []
    manifest: list[dict] = []

    for path in tqdm(image_files, desc="Labeling"):
        res = pipeline.process(path, labels_dir)
        if not res.ok:
            failed.append((path.name, res.error or "unknown"))
            continue
        total_labels += res.n_labels
        total_detections += res.n_detections
        manifest.append({
            "image": str(path),
            "label": str(res.label_path) if res.label_path else None,
            "n_labels": res.n_labels,
            "n_detections": res.n_detections,
            "per_label": res.per_label,
        })

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.write_dataset_yaml:
        write_dataset_yaml(output_dir, list(FRUIT_CLASSES))

    print("\n" + "=" * 60)
    print(f"processed    : {len(image_files) - len(failed)}/{len(image_files)}")
    print(f"detections   : {total_detections}")
    print(f"labels kept  : {total_labels}")
    print(f"manifest     : {manifest_path}")
    if failed:
        print(f"failed ({len(failed)}):")
        for name, err in failed:
            print(f"  - {name}: {err}")
    print("=" * 60)


if __name__ == "__main__":
    main()
