"""Session + image upload endpoints."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend import config
from backend.schemas import (
    ImageLabels,
    ImageUploadResponse,
    LabelEntry,
    SessionCreateResponse,
    SessionStats,
)
from backend.services.session_store import store
from utils.image_utils import load_image


router = APIRouter(prefix="/session", tags=["session"])

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@router.post("", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    session = store.create_session()
    return SessionCreateResponse(session_id=session.session_id)


@router.post("/{session_id}/images", response_model=ImageUploadResponse)
async def upload_images(session_id: str, files: list[UploadFile] = File(...)) -> ImageUploadResponse:
    try:
        session = store.require_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    image_ids: list[str] = []
    for up in files:
        suffix = Path(up.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            continue
        dest = session.images_dir / (Path(up.filename).name)
        with dest.open("wb") as f:
            shutil.copyfileobj(up.file, f)
        try:
            img = load_image(str(dest))
        except Exception:
            dest.unlink(missing_ok=True)
            continue
        rec = store.add_image(session_id, dest, img.shape[1], img.shape[0])
        image_ids.append(rec.image_id)

    return ImageUploadResponse(session_id=session_id, image_ids=image_ids)


@router.get("/{session_id}/images")
def list_images(session_id: str, sort_by_priority: bool = False) -> list[dict]:
    try:
        images = store.list_images(session_id, sort_by_priority=sort_by_priority)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return [
        {
            "image_id": r.image_id,
            "filename": r.path.name,
            "width": r.width,
            "height": r.height,
            "n_labels": len(r.labels),
            "uncertain": r.uncertain,
            "reviewed": r.reviewed,
            "priority": r.priority,
        }
        for r in images
    ]


@router.get("/{session_id}/images/{image_id}/file")
def get_image_file(session_id: str, image_id: str):
    try:
        rec = store.get_image(session_id, image_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(rec.path)


@router.get("/{session_id}/images/{image_id}/labels", response_model=ImageLabels)
def get_image_labels(session_id: str, image_id: str) -> ImageLabels:
    try:
        rec = store.get_image(session_id, image_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="image not found")
    return ImageLabels(
        session_id=session_id,
        image_id=image_id,
        width=rec.width,
        height=rec.height,
        labels=[LabelEntry(**lbl) for lbl in rec.labels],
    )


@router.post("/{session_id}/images/{image_id}/labels", response_model=ImageLabels)
def set_image_labels(session_id: str, image_id: str, payload: ImageLabels) -> ImageLabels:
    try:
        rec = store.get_image(session_id, image_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="image not found")
    store.set_labels(
        session_id, image_id,
        [lbl.model_dump() for lbl in payload.labels],
    )
    rec.reviewed = True
    return get_image_labels(session_id, image_id)


@router.get("/{session_id}/stats", response_model=SessionStats)
def session_stats(session_id: str) -> SessionStats:
    try:
        images = store.list_images(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    labeled = sum(1 for r in images if r.labels)
    total_labels = sum(len(r.labels) for r in images)
    dist: dict[str, int] = {cls: 0 for cls in config.DOMAIN_CLASSES}
    for r in images:
        for lbl in r.labels:
            name = lbl.get("cls_name") or config.DOMAIN_CLASSES[lbl["cls_id"]]
            dist[name] = dist.get(name, 0) + 1
    uncertain = sum(1 for r in images if r.uncertain)
    return SessionStats(
        session_id=session_id,
        image_count=len(images),
        labeled_count=labeled,
        total_labels=total_labels,
        class_distribution=dist,
        uncertain_queue=uncertain,
    )
