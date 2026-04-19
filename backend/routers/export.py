"""YOLO-seg export endpoint: zip all labels + dataset.yaml."""

from __future__ import annotations

import io
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend import config
from backend.services.session_store import store
from utils.export import write_dataset_yaml


router = APIRouter(prefix="/export", tags=["export"])


@router.get("/{session_id}")
def export_session(session_id: str, format: str = "yolo-seg") -> StreamingResponse:
    if format != "yolo-seg":
        raise HTTPException(status_code=400, detail="only yolo-seg is supported")
    try:
        session = store.require_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    # Write dataset.yaml fresh into the session root before zipping
    yaml_path = write_dataset_yaml(session.root, config.DOMAIN_CLASSES)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(yaml_path, arcname="dataset.yaml")
        for rec in store.list_images(session_id):
            stem = rec.path.stem
            # Serialize current in-memory labels (overrides the on-disk copy if edited)
            txt_lines = []
            for lbl in rec.labels:
                cls_id = int(lbl["cls_id"])
                poly = lbl.get("polygon") or []
                if len(poly) < 6:
                    continue
                coords = " ".join(f"{v:.6f}" for v in poly)
                txt_lines.append(f"{cls_id} {coords}")
            zf.writestr(f"labels/{stem}.txt", "\n".join(txt_lines) + ("\n" if txt_lines else ""))

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{session_id}_yolo_seg.zip"'},
    )
