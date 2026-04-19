"""Auto-labeling pipeline execution + mask refinement endpoints."""

from __future__ import annotations

import asyncio
import json

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from backend import config
from backend.schemas import RefineRequest, RefineResponse, RunRequest, RunResponse
from backend.services.detector_service import get_detector
from backend.services.sam_service import get_sam
from backend.services.session_store import store
from utils.export import mask_to_yolo_polygon
from utils.image_utils import load_image, resize_image
from utils.label_pipeline import FruitLabelPipeline
from utils.uncertainty import (
    UncertaintyThresholds,
    image_priority,
    label_uncertainty,
)


router = APIRouter(tags=["pipeline"])


@router.post("/pipeline/{session_id}/run", response_model=RunResponse)
async def run_pipeline(session_id: str, req: RunRequest) -> RunResponse:
    try:
        session = store.require_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    images = store.list_images(session_id)
    if not images:
        raise HTTPException(status_code=400, detail="no images uploaded")

    task = store.create_task(session_id, total=len(images))
    loop = asyncio.get_running_loop()
    asyncio.create_task(_execute_pipeline(loop, session, images, task.task_id, req))
    return RunResponse(task_id=task.task_id)


async def _execute_pipeline(loop, session, images, task_id, req: RunRequest) -> None:
    task = store.get_task(task_id)
    if task is None:
        return
    try:
        sam = await loop.run_in_executor(None, get_sam, req.sam_model)
        detector = await loop.run_in_executor(None, get_detector, req.det_conf)
        pipeline = FruitLabelPipeline(
            sam=sam,
            detector=detector,
            epsilon=req.epsilon,
            max_image_size=req.max_image_size,
            multi_contour=req.multi_contour,
            min_area_ratio=req.min_area_ratio,
        )

        for i, rec in enumerate(images):
            task.current_image = rec.path.name
            result = await loop.run_in_executor(
                None, pipeline.process, rec.path, session.labels_dir
            )
            thresholds = UncertaintyThresholds()
            entries: list[dict] = []
            uncertain_count = 0
            for item in result.per_label:
                if item.get("error"):
                    continue
                polygons = item.get("polygons")
                if not polygons:
                    continue
                is_uncertain, _ = label_uncertainty(
                    sam_scores=item.get("sam_scores"),
                    det_conf=item.get("conf"),
                    thresholds=thresholds,
                )
                if is_uncertain:
                    uncertain_count += 1
                # One detection may yield multiple disjoint polygons under
                # multi_contour=True; expand each as its own session entry.
                for poly in polygons:
                    entries.append({
                        "cls_id": item["cls_id"],
                        "cls_name": item["cls_name"],
                        "polygon": poly,
                        "sam_score": item.get("sam_score"),
                        "det_conf": item.get("conf"),
                        "bbox": item.get("bbox"),
                    })
            uncertain = uncertain_count > 0
            rec_priority = image_priority(uncertain_count, len(entries), uncertain)
            rec.priority = rec_priority  # stored on ImageRecord via dynamic attr
            store.set_labels(session.session_id, rec.image_id, entries)
            store.mark_uncertain(session.session_id, rec.image_id, uncertain)

            task.done = i + 1
            if uncertain:
                task.uncertain_count += 1
            await store.publish(task_id, {
                "task_id": task_id,
                "done": task.done,
                "total": task.total,
                "current_image": task.current_image,
                "uncertain_count": task.uncertain_count,
                "status": "running",
            })

        task.status = "completed"
        await store.publish(task_id, {
            "task_id": task_id,
            "done": task.done,
            "total": task.total,
            "current_image": None,
            "uncertain_count": task.uncertain_count,
            "status": "completed",
        })
    except Exception as exc:
        task.status = "failed"
        task.error = str(exc)
        await store.publish(task_id, {
            "task_id": task_id,
            "done": task.done,
            "total": task.total,
            "uncertain_count": task.uncertain_count,
            "status": "failed",
            "error": str(exc),
        })


@router.websocket("/progress/{task_id}")
async def progress_ws(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    task = store.get_task(task_id)
    if task is None:
        await websocket.send_text(json.dumps({"error": "unknown task_id"}))
        await websocket.close()
        return

    queue = store.subscribe(task_id)
    try:
        await websocket.send_text(json.dumps({
            "task_id": task_id,
            "done": task.done,
            "total": task.total,
            "uncertain_count": task.uncertain_count,
            "status": task.status,
        }))
        while True:
            payload = await queue.get()
            await websocket.send_text(json.dumps(payload))
            if payload.get("status") in ("completed", "failed"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        store.unsubscribe(task_id, queue)


@router.post("/refine", response_model=RefineResponse)
async def refine_mask(req: RefineRequest) -> RefineResponse:
    try:
        rec = store.get_image(req.session_id, req.image_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="image not found")

    loop = asyncio.get_running_loop()
    sam = await loop.run_in_executor(None, get_sam, config.DEFAULT_SAM_MODEL)

    image = load_image(str(rec.path))
    image = resize_image(image, max_size=config.DEFAULT_MAX_IMAGE_SIZE)
    h, w = image.shape[:2]

    def _run() -> tuple[np.ndarray, float]:
        sam.set_image(image)
        if req.points:
            masks, scores, _ = sam.predict_with_points(req.points, req.point_labels)
        elif req.bbox is not None:
            masks, scores, _ = sam.predict_with_box(req.bbox)
        else:
            raise ValueError("either points or bbox required")
        best_mask, best_score = sam.get_best_mask(masks, scores)
        return best_mask, float(best_score)

    best_mask, best_score = await loop.run_in_executor(None, _run)
    polygon = mask_to_yolo_polygon(best_mask, w, h, epsilon=config.DEFAULT_EPSILON)
    if polygon is None:
        raise HTTPException(status_code=422, detail="mask produced empty polygon")
    return RefineResponse(polygon=polygon, sam_score=best_score)
