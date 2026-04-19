# Architecture

SAM Auto-Label Studio의 시스템 구조 문서. 컴포넌트, 책임 분리, 데이터 흐름, 상태 관리 모델을 정리.

> UML 다이어그램(클래스/시퀀스)은 [`uml.md`](./uml.md) 참고.

---

## 1. 전체 개요

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Browser (React + Vite)                          │
│   AutoLabelTab    ReviewQueueTab    ExportTab     MaskCanvas (Canvas)    │
│        │                │                │                               │
│        └────────────────┴────────────────┴────► Zustand store (5 keys)   │
└────────────────────────────────┬─────────────────────────────────────────┘
                          HTTP REST + WebSocket
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│                       FastAPI Backend (uvicorn)                          │
│  routers/                services/                                       │
│   ├ session              ├ session_store (in-memory + filesystem)        │
│   ├ pipeline (+WS)       ├ detector_service (FruitDetector singleton)    │
│   └ export               └ sam_service (SAMPredictor cache)              │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│                       ML / Pipeline core (utils/)                        │
│  FruitDetector ──► SAMPredictor ──► mask_to_yolo_polygon                 │
│   (YOLOv8s)        (ViT-B / ViT-H)   (cv2.approxPolyDP)                  │
│                                                                          │
│  FruitLabelPipeline 가 위 흐름을 묶어 process(image) → LabelResult       │
│  uncertainty.label_uncertainty / image_priority — 큐 정렬용 점수         │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│  Data layer (filesystem)                                                 │
│   models/   SAM 체크포인트 (sam_vit_b/h.pth)                             │
│   runs/     YOLOv8 학습 산출물 (weights, 로그, 그래프)                   │
│   datasets/ 학습용 데이터셋 (fruit10_merged 등)                          │
│   outputs/sessions/<sid>/  세션별 이미지/라벨 (영구 저장)                │
│   outputs/eval/            eval_iou JSON 리포트                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 레이어 책임

### 2.1 Frontend (`frontend/src/`)
- **`App.tsx`**: 마운트 시 `health()` + `createSession()` 호출 → `sessionId` 확보 후 탭 라우팅.
- **Tabs**:
  - `AutoLabelTab` — 업로드 / 파라미터 (sam_model, det_conf, multi_contour) / 학습 시작 / WS 진행률 표시.
  - `ReviewQueueTab` — 우선순위 정렬된 이미지 목록, 마스크 보정 (point/box prompt), 클래스 변경, 저장.
  - `ExportTab` — 통계 요약, YOLO-seg zip 다운로드.
- **`MaskCanvas`** — `<canvas>` 위에 polygon 그리기, 점/박스 prompt 입력 처리.
- **`useTaskProgress(taskId)`** — `/ws/progress/{taskId}` WebSocket 구독, ProgressEvent 스트림.
- **Zustand store (`store.ts`)** — `sessionId`, `tab`, `classes`, `stats`, `selectedImageId` 5개 키만 글로벌. 나머지는 각 탭의 로컬 state.

### 2.2 Backend (`backend/`)
- **Routers** (`backend/routers/`)
  - `session.py` — 세션 생성, 이미지 업로드/조회, 라벨 GET/POST, 통계.
  - `pipeline.py` — `POST /pipeline/{sid}/run` (background asyncio task), `WS /progress/{tid}`, `POST /refine`.
  - `export.py` — `GET /export/{sid}` → 메모리 zip stream.
- **Services** (`backend/services/`)
  - `session_store.py` — **단일 인스턴스 in-memory store**. `Session`, `ImageRecord`, `Task` 데이터 클래스 보유. `threading.Lock`로 동시성 보호. WebSocket 구독은 task별 `asyncio.Queue` 리스트로 fan-out.
  - `detector_service.py` — `FruitDetector`를 lazy 싱글턴으로 보관. `conf` 변경 시 재로드.
  - `sam_service.py` — model_type별 (`vit_b` / `vit_h`) `SAMPredictor` 캐시.
- **Schemas** (`backend/schemas.py`) — Pydantic DTO. 라우터 ↔ 클라이언트 계약.
- **Config** (`backend/config.py`) — 디렉터리 경로, SAM 체크포인트 매핑, 도메인 클래스 (`FRUIT_CLASSES`), CORS 화이트리스트.

### 2.3 ML / Pipeline core (`utils/`)
- **`FruitDetector`** — Ultralytics YOLO 래퍼. `detect(image) → list[dict(bbox, cls_id, cls_name, conf)]`.
- **`SAMPredictor`** — segment-anything 래퍼. `set_image / predict_with_box / predict_with_points / predict_with_points_and_box / get_best_mask / generate_masks`.
- **`FruitLabelPipeline`** — `detect → set_image → predict_with_box → get_best_mask → mask_to_yolo_polygon → write_yolo_seg` 오케스트레이션.
- **`mask_to_yolo_polygon`** (`export.py`) — 컨투어 → `cv2.approxPolyDP(epsilon=0.005)` → 정규화된 좌표.
- **`uncertainty.py`** — `label_uncertainty(sam_scores, det_conf, thresholds)`, `image_priority(...)`. 라벨링 후 큐 정렬에 사용.
- **`image_utils.py`** — `load_image / resize_image / save_image / get_mask_bbox` 등 보조.
- **`visualization.py`** — matplotlib 기반 디버그 헬퍼 (런타임 의존성 없음, 노트북/스크립트용).

### 2.4 Scripts / training (`scripts/`)
- 런타임 파이프라인과 분리된 **detector 재학습** 도구.
- 순서: `remap_henningheyen` / `remap_deepnir_mango` → `merge_fruit_datasets` → `train_fruit_detector` → 결과 weights를 `utils/detector.py::DEFAULT_FRUIT_WEIGHTS`로 연결.

### 2.5 CLI 진입점
- `batch_process.py` — 디렉터리 단위 자동 라벨링. `FruitLabelPipeline` 직접 사용. manifest.json에 sam/det 점수 기록.
- `webcam_demo.py` — SAM `generate_masks` (auto-mask) + OpenCV 윈도우. detector 미사용.
- `tests/eval_iou.py` — detector + SAM 파이프라인 IoU 평가 (greedy match).

---

## 3. 데이터 흐름

### 3.1 자동 라벨링 (Auto Labeling)

```
User → AutoLabelTab
  ├─ uploadImages(files)        → POST /session/{sid}/images        (multipart)
  │   └ session_store.add_image(sid, path, w, h)
  └─ runPipeline({params})      → POST /pipeline/{sid}/run
      │   └ store.create_task(sid, total) → task_id 반환
      │   └ asyncio.create_task(_execute_pipeline)
      └ useTaskProgress(taskId) → WS /progress/{taskId}

_execute_pipeline (async background):
  load SAM (sam_service.get_sam)
  load detector (detector_service.get_detector)
  pipeline = FruitLabelPipeline(sam, detector, ...)
  for img in store.list_images(sid):
      result = pipeline.process(img.path, labels_dir)
      for each label: label_uncertainty(...) → uncertain 여부
      store.set_labels(sid, image_id, labels)
      store.mark_uncertain(...)
      rec.priority = image_priority(...)
      store.publish(task_id, ProgressEvent(done, total, current_image, ...))
  store.publish(... status="completed")

Frontend: WS 메시지 수신 → setEvent → 5 tick마다 listImages 재호출 → 그리드 갱신
```

### 3.2 마스크 보정 (Refine)

```
ReviewQueueTab → MaskCanvas 좌클릭/우클릭 (point) 또는 드래그 (box)
  → refineMask({session_id, image_id, points, point_labels} or {bbox})
     POST /refine
       load_image → resize_image
       sam.set_image
       sam.predict_with_points(...) 또는 sam.predict_with_box(...)
       sam.get_best_mask
       mask_to_yolo_polygon(mask, w, h, epsilon=0.005)
       return {polygon, sam_score}
  → labels 배열 in-place 수정 → setDirty(true)
  → onSave → POST /session/{sid}/images/{iid}/labels (전체 labels 배열)
       store.set_labels(...) — 영구 저장은 export 시점
```

### 3.3 Export

```
ExportTab → getStats(sid)            → GET /session/{sid}/stats
        → "Download" 클릭            → GET /export/{sid}?format=yolo-seg
            store.list_images(sid) 순회
            각 ImageRecord.labels → "cls x1 y1 x2 y2 ... xn yn" 라인
            zipfile에 labels/{stem}.txt 추가
            dataset.yaml (DOMAIN_CLASSES) 추가
            StreamingResponse(zip_buf, media_type="application/zip")
```

---

## 4. 상태 관리

### 4.1 Backend (in-memory)
| Entity | 위치 | 휘발성 |
|---|---|---|
| `Session` | `SessionStore._sessions[sid]` | **메모리** (서버 재시작 시 소실). 단, 이미지 파일 자체는 `outputs/sessions/<sid>/images/`에 디스크 저장 |
| `ImageRecord` | `Session.images[image_id]` | 메모리. labels는 in-memory list[dict], export 시 디스크에 zip으로만 출력 |
| `Task` | `SessionStore._tasks[tid]` | 메모리. completed 후에도 즉시 GC 안 함 |
| WS subscribers | `SessionStore._subscribers[tid]` | 메모리. 연결 종료 시 unsubscribe |

> **시사점**: 라벨은 `POST /session/.../labels` 후에도 zip export 시점까지 디스크에 쓰이지 않음. 서버 프로세스 재시작 시 라벨 손실 가능 — 운영 환경 적용 시 영속화 계층 추가 필요 (TODO).

### 4.2 Frontend (Zustand + 로컬)
- 글로벌(Zustand): `sessionId`, `tab`, `classes`, `stats`, `selectedImageId` 만.
- 그 외 (`images`, `labels`, `samModel`, `detConf`, `pendingPoints`, ...)는 모두 각 탭 컴포넌트의 `useState` 로컬.
- 탭 전환 시 로컬 state 소실 의도: Auto/Review/Export 각각이 마운트 시 서버에서 다시 fetch.

---

## 5. 설정 & 도메인

### 5.1 `backend/config.py` 핵심
- `DOMAIN_NAME = "fruit"`, `DOMAIN_CLASSES = (...10 fruits)`, `DOMAIN_DET_WEIGHTS = utils.detector.DEFAULT_FRUIT_WEIGHTS`
- `SAM_CHECKPOINTS = {"vit_b": models/sam_vit_b.pth, "vit_h": models/sam_vit_h.pth}`
- 기본값: `DEFAULT_SAM_MODEL=vit_b`, `DEFAULT_DET_CONF=0.15`, `DEFAULT_MAX_IMAGE_SIZE=1024`, `DEFAULT_EPSILON=0.005`
- `ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]`

### 5.2 `utils/detector.py` 핵심
- `DEFAULT_FRUIT_WEIGHTS` → `runs/detect/runs/detect/fruit10_merged_v2b_imgsz1280/weights/best.pt`
- 기본 `imgsz=640` (학습은 1280) — 추론 정렬 필요 시 `to-do-work.md` TODO 3 참고.

### 5.3 도메인 교체 시
- `DOMAIN_CLASSES` + `DOMAIN_DET_WEIGHTS` 수정 → schema는 `cls_id` (int) 기반이라 클라 영향 최소.
- `FruitDetector.CLASSES` 가 `FRUIT_CLASSES`에 직접 묶여 있어 클래스 추가/삭제 시 같이 갱신 필요.

---

## 6. 동시성 모델

- `SessionStore` — `threading.Lock`으로 dict 변경 보호. WS publish는 비동기 큐, 락 안에서 호출되어도 안전 (큐 put nowait).
- 파이프라인 실행은 `asyncio.create_task` 백그라운드. 무거운 detect/SAM 호출은 `loop.run_in_executor`로 thread offload (uvicorn 단일 프로세스 가정).
- SAM/YOLO 모델 인스턴스는 service 모듈의 lazy 싱글턴 — 여러 동시 요청이 같은 모델을 공유. predict 호출은 thread-safe하지 않을 수 있어 직렬화 (실제 production 환경에서는 모델별 lock 추가 검토).

---

## 7. 배포 / 실행 토폴로지

```
Dev
  - frontend: vite dev server (http://localhost:5173) — Axios가 /api → 8000 프록시
  - backend:  uvicorn --reload (http://127.0.0.1:8000)

Prod (현재 미구성, 권장 안)
  - frontend: vite build → 정적 파일을 nginx 또는 FastAPI StaticFiles로 서빙
  - backend:  uvicorn (gunicorn + uvicorn workers) 단일 프로세스 권장 (in-memory store 때문)
              영속화 추가 시 worker 다중화 가능
```

GPU 1장 사용. 동시 다중 세션은 GPU 메모리 경쟁 (vit_h + YOLO 합 ~3GB). 8GB GPU에서 vit_b + YOLO는 여유 있음.

---

## 8. 확장 / 변경 시 주의

- **새 도메인 추가** — `config.DOMAIN_CLASSES`, `DOMAIN_DET_WEIGHTS`, `FRUIT_CLASSES` 동시 갱신. 프론트는 `health()`로 클래스 받아오므로 코드 변경 불필요.
- **추론 imgsz 변경** — `FruitDetector(__init__)`의 `imgsz` 인자 또는 service 호출부에서 명시.
- **새 prompt 모드 추가** — `SAMPredictor` 메서드 추가 → `pipeline.py::refine_mask()` 분기 추가 → 클라 `MaskCanvas` 인터랙션 추가.
- **라벨 영속화** — `session_store.set_labels` 시점에 disk write 추가하거나 SQLite/Postgres로 store 전환.
