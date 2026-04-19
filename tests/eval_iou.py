"""
End-to-end label quality evaluation against the yolov8_fruit_detection val set.

The GT labels are YOLO **detection** boxes (cls cx cy w h, normalized), so we
report two complementary metrics:

  1. bbox-IoU between predicted polygon's tight bbox and matched GT box
     (per class greedy match by IoU). Captures localization error introduced
     by detector + SAM box prompt + polygon simplification.

  2. mask self-consistency: rasterize our polygon back to a binary mask and
     compare against the raw SAM "best mask" (pre-polygon). This isolates
     the loss caused by approxPolyDP at the chosen epsilon.

Usage:
    python -m tests.eval_iou \\
        --val-dir /home/duck/PycharmProjects/yolov8_fruit_detection/data/val \\
        --max-images 50 \\
        --epsilons 0.001,0.005,0.01
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.detector import FruitDetector, FRUIT_CLASSES
from utils.export import mask_to_yolo_polygon
from utils.image_utils import load_image, resize_image
from utils.sam_predictor import SAMPredictor


@dataclass
class GTBox:
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def load_gt(label_path: Path, img_w: int, img_h: int) -> list[GTBox]:
    if not label_path.exists():
        return []
    boxes: list[GTBox] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append(GTBox(cls_id, x1, y1, x2, y2))
    return boxes


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def polygon_to_bbox(polygon_norm: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    pts = np.array(polygon_norm, dtype=np.float64).reshape(-1, 2)
    xs = pts[:, 0] * img_w
    ys = pts[:, 1] * img_h
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def polygon_to_mask(polygon_norm: list[float], img_w: int, img_h: int) -> np.ndarray:
    pts = np.array(polygon_norm, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    return float(inter / union) if union > 0 else 0.0


def greedy_match(
    preds: list[tuple[int, tuple[float, float, float, float], int]],
    gts: list[GTBox],
    iou_thr: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """
    Match preds to gts by class, greedy by IoU descending.
    Returns (matches, matched_pred_indices, matched_gt_indices).
    """
    candidates: list[tuple[float, int, int]] = []
    for pi, (cls_id, pbox, _) in enumerate(preds):
        for gi, gt in enumerate(gts):
            if gt.cls_id != cls_id:
                continue
            iou = bbox_iou(pbox, (gt.x1, gt.y1, gt.x2, gt.y2))
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True)

    matched: list[tuple[int, int, float]] = []
    used_p: set[int] = set()
    used_g: set[int] = set()
    for iou, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        matched.append((pi, gi, iou))
        used_p.add(pi)
        used_g.add(gi)
    return matched, used_p, used_g


def evaluate(
    val_dir: Path,
    max_images: int,
    epsilons: list[float],
    sam_model: str,
    det_conf: float,
    max_image_size: int,
    iou_thr: float,
    weights: Path | None = None,
) -> dict:
    image_dir = val_dir / "images"
    label_dir = val_dir / "labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    images = images[:max_images]

    sam = SAMPredictor(model_type=sam_model)
    detector_kwargs = {"conf": det_conf}
    if weights is not None:
        detector_kwargs["weights"] = weights
    detector = FruitDetector(**detector_kwargs)

    total_gt = 0
    total_pred = 0
    total_tp = 0
    bbox_ious: list[float] = []
    per_class_iou: dict[int, list[float]] = defaultdict(list)
    per_class_tp: dict[int, int] = defaultdict(int)
    per_class_pred: dict[int, int] = defaultdict(int)
    per_class_gt: dict[int, int] = defaultdict(int)

    epsilon_self_iou: dict[float, list[float]] = {e: [] for e in epsilons}

    t0 = time.time()
    for idx, img_path in enumerate(images):
        try:
            image = load_image(str(img_path))
        except Exception as e:
            print(f"[skip] {img_path.name}: {e}")
            continue
        image = resize_image(image, max_size=max_image_size)
        h, w = image.shape[:2]

        gt_boxes = load_gt(label_dir / f"{img_path.stem}.txt", w, h)
        for gt in gt_boxes:
            per_class_gt[gt.cls_id] += 1
        total_gt += len(gt_boxes)

        detections = detector.detect(image)
        if not detections:
            continue

        sam.set_image(image)
        preds: list[tuple[int, tuple[float, float, float, float], int]] = []
        for det in detections:
            bbox = det["bbox"]
            cls_id = det["cls_id"]
            try:
                masks, scores, _ = sam.predict_with_box(bbox)
                best_mask, _ = sam.get_best_mask(masks, scores)
            except Exception:
                continue
            polygon = mask_to_yolo_polygon(best_mask, w, h, epsilon=0.005)
            if polygon is None:
                continue
            pbox = polygon_to_bbox(polygon, w, h)
            preds.append((cls_id, pbox, len(preds)))
            per_class_pred[cls_id] += 1
            total_pred += 1

            for eps in epsilons:
                eps_poly = mask_to_yolo_polygon(best_mask, w, h, epsilon=eps)
                if eps_poly is None:
                    continue
                eps_mask = polygon_to_mask(eps_poly, w, h)
                epsilon_self_iou[eps].append(mask_iou(best_mask, eps_mask))

        matches, _, _ = greedy_match(preds, gt_boxes, iou_thr)
        for pi, _, iou in matches:
            cls_id = preds[pi][0]
            bbox_ious.append(iou)
            per_class_iou[cls_id].append(iou)
            per_class_tp[cls_id] += 1
            total_tp += 1

        if (idx + 1) % 10 == 0:
            print(f"  processed {idx + 1}/{len(images)}")

    elapsed = time.time() - t0

    def safe_mean(xs: list[float]) -> float:
        return statistics.mean(xs) if xs else 0.0

    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gt if total_gt else 0.0

    per_class = {}
    for cls_id in sorted(set(per_class_pred) | set(per_class_gt)):
        tp = per_class_tp[cls_id]
        n_pred = per_class_pred[cls_id]
        n_gt = per_class_gt[cls_id]
        per_class[FRUIT_CLASSES[cls_id]] = {
            "n_gt": n_gt,
            "n_pred": n_pred,
            "tp": tp,
            "precision": tp / n_pred if n_pred else 0.0,
            "recall": tp / n_gt if n_gt else 0.0,
            "mean_bbox_iou": safe_mean(per_class_iou[cls_id]),
        }

    epsilon_summary = {
        f"{eps:.4f}": {
            "n": len(samples),
            "mean_self_iou": safe_mean(samples),
            "min_self_iou": min(samples) if samples else 0.0,
        }
        for eps, samples in epsilon_self_iou.items()
    }

    return {
        "n_images": len(images),
        "n_gt": total_gt,
        "n_pred": total_pred,
        "tp": total_tp,
        "precision_at_iou": {f"{iou_thr:.2f}": precision},
        "recall_at_iou": {f"{iou_thr:.2f}": recall},
        "mean_bbox_iou_matched": safe_mean(bbox_ious),
        "per_class": per_class,
        "epsilon_self_consistency": epsilon_summary,
        "elapsed_sec": elapsed,
        "config": {
            "sam_model": sam_model,
            "det_conf": det_conf,
            "max_image_size": max_image_size,
            "iou_threshold": iou_thr,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("/home/duck/PycharmProjects/yolov8_fruit_detection/data/val"),
    )
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--epsilons", type=str, default="0.001,0.005,0.01")
    parser.add_argument("--sam-model", type=str, default="vit_b")
    parser.add_argument("--det-conf", type=float, default=0.15)
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional JSON output path")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Override FruitDetector weights path (default: utils/detector.py DEFAULT_FRUIT_WEIGHTS)")
    args = parser.parse_args()

    epsilons = [float(x) for x in args.epsilons.split(",") if x.strip()]
    report = evaluate(
        val_dir=args.val_dir,
        max_images=args.max_images,
        epsilons=epsilons,
        sam_model=args.sam_model,
        det_conf=args.det_conf,
        max_image_size=args.max_image_size,
        iou_thr=args.iou_threshold,
        weights=args.weights,
    )

    print("\n=== Summary ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
