"""
Train YOLOv8 on the merged 10-class fruit dataset.

By default starts from `yolov8s.pt` (mirrors the original fruit_detection
training) and writes weights to `runs/detect/fruit10_merged/weights/best.pt`.
After training, the best weights can be wired into utils/detector.py via
DEFAULT_FRUIT_WEIGHTS or passed at runtime.

Usage (from project root):
    python -m scripts.train_fruit_detector \\
        --data datasets/fruit10_merged/dataset.yaml \\
        --epochs 60 --batch 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=Path("datasets/fruit10_merged/dataset.yaml"))
    ap.add_argument("--model", type=str, default="yolov8s.pt",
                    help="Pretrained YOLO weights to fine-tune from")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--name", type=str, default="fruit10_merged")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        save=True,
        save_period=10,
    )
    print(f"Training done. Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
