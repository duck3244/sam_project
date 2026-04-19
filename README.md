# SAM Auto-Label Studio

YOLOv8 detector + Meta Segment Anything (SAM1) 기반 과일 도메인 자동 라벨링 파이프라인.
사전학습된 fruit detector(10 classes)로 bounding box를 잡고, SAM box-prompt로 인스턴스 마스크를 생성해 YOLO-seg 포맷으로 export합니다. CLI 배치 또는 React + FastAPI 기반 웹 UI로 사용할 수 있습니다.

## 아키텍처

```
┌────────────────────────────┐   HTTP/WS   ┌──────────────────────────┐
│  React + Vite + Tailwind   │ ◀─────────▶ │  FastAPI Backend         │
│  - Auto Labeling           │             │  - Session store         │
│  - Review Queue (Canvas)   │             │  - Pipeline orchestrator │
│  - Export                  │             │  - YOLO-seg export       │
└────────────────────────────┘             └────────────┬─────────────┘
                                                        │
             ┌──────────────────────┬───────────────────┼────────────────────┐
             ▼                      ▼                   ▼                    ▼
      FruitDetector (YOLOv8)  SAMPredictor (ViT-B/H)  Uncertainty     Export (YOLO-seg)
      best.pt · 10 classes    box/point prompt        (score + var)   polygon txt + yaml
```

## 핵심 흐름

1. **Detection** — `FruitDetector`가 이미지에서 10 fruit class의 bbox 생성.
2. **Segmentation** — 각 bbox를 SAM `predict_with_box`에 주입 → best mask 선택.
3. **Polygonization** — `mask_to_yolo_polygon`이 `cv2.approxPolyDP(epsilon=0.005)`로 컨투어 단순화 + 정규화.
4. **Uncertainty** — multimask score 분산 + best score + det conf 기반 우선순위.
5. **Export** — YOLO-seg `.txt` + `dataset.yaml`을 zip으로 묶어 다운로드.

## 디렉토리 구조

```
sam_project/
├── backend/                 # FastAPI (routers/, services/, schemas.py)
├── frontend/                # React + Vite (src/tabs, src/components, ...)
├── utils/
│   ├── detector.py          # FruitDetector (YOLOv8, default v2b weights)
│   ├── sam_predictor.py     # SAM wrapper
│   ├── label_pipeline.py    # detect → SAM → export 오케스트레이터
│   ├── export.py            # mask → YOLO-seg polygon
│   ├── uncertainty.py       # multimask 기반 우선순위
│   └── image_utils.py
├── scripts/                 # detector 재학습 파이프라인
│   ├── merge_fruit_datasets.py
│   ├── remap_deepnir_mango.py
│   ├── remap_henningheyen.py
│   └── train_fruit_detector.py
├── tests/                   # pytest (export, uncertainty, eval_iou)
├── datasets/                # 학습용 데이터 (fruit10_merged 등)
├── runs/                    # YOLOv8 학습 결과 (weights, 로그)
├── outputs/                 # 라벨링/평가 산출물
├── batch_process.py         # CLI 진입점
├── download_model.py        # SAM 체크포인트 다운로드
├── requirements.txt
├── to-do-work.md            # 남은 작업 메모 (detector 개선, 데이터 보강 등)
└── README.md
```

## 빠른 시작

### 1. 환경 준비

```bash
pip install -r requirements.txt
```

주요 의존성: `ultralytics`, `segment-anything`, `fastapi`, `uvicorn[standard]`, `opencv-python`, `shapely`, `pyyaml`.

### 2. 모델 체크포인트

```bash
# SAM (둘 중 하나)
python download_model.py --model vit_b   # 375MB, 빠름 (default)
python download_model.py --model vit_h   # 2.4GB, 정확

# YOLO fruit detector (v2b, imgsz=1280으로 학습된 best.pt)
# utils/detector.py의 DEFAULT_FRUIT_WEIGHTS가 가리키는 경로:
#   runs/detect/runs/detect/fruit10_merged_v2b_imgsz1280/weights/best.pt
# 이 파일이 없으면 학습 또는 다른 weights 경로 지정 필요
```

10 classes: `apple, banana, orange, strawberry, grape, pineapple, watermelon, mango, peach, cherry`.

### 3. CLI 배치 라벨링

```bash
python batch_process.py \
  --input_dir /path/to/images \
  --output_dir ./outputs/run1 \
  --sam_model vit_b \
  --det_conf 0.3 \
  --write_dataset_yaml
```

산출물:
- `outputs/run1/labels/*.txt` — 이미지당 YOLO-seg 라벨
- `outputs/run1/dataset.yaml` — Ultralytics 학습용
- `outputs/run1/manifest.json` — detection / SAM 점수 기록

### 4. 웹 UI

```bash
# Backend
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
# OpenAPI: http://127.0.0.1:8000/docs

# Frontend
cd frontend
npm install
npm run dev        # http://localhost:5173
# 또는
npm run build      # dist/ 정적 산출물
```

탭 구성:
1. **Auto Labeling** — 드래그&드롭 업로드 → SAM 모델 선택 → WebSocket 진행률
2. **Review Queue** — 마스크 보정 (클래스 변경 / 포인트 재분할 / 삭제 / 저장), 우선순위 정렬
3. **Export** — 클래스 분포 차트 + `yolo-seg.zip` 다운로드

## 주요 API

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/session` | 새 세션 생성 |
| `POST` | `/session/{sid}/images` | multipart 이미지 업로드 |
| `GET` | `/session/{sid}/images` | 이미지 목록 (`sort_by_priority` 옵션) |
| `GET` | `/session/{sid}/images/{iid}/labels` | 현재 라벨 조회 |
| `POST` | `/session/{sid}/images/{iid}/labels` | 수정된 라벨 저장 |
| `POST` | `/pipeline/{sid}/run` | 자동 라벨링 태스크 시작 |
| `WS` | `/progress/{tid}` | 진행률 스트리밍 |
| `POST` | `/refine` | 포인트 프롬프트 기반 마스크 재분할 |
| `GET` | `/export/{sid}` | YOLO-seg zip 다운로드 |

## Detector 재학습

10-class fruit detector는 `scripts/`에서 직접 재학습할 수 있습니다.

```bash
# 1. 데이터 소스 정규화 (필요 시)
python -m scripts.remap_henningheyen --src ... --dst datasets/henningheyen_fruit10
python -m scripts.remap_deepnir_mango --src ... --dst datasets/deepNIR_mango --mango-cls 6

# 2. 통합 데이터셋 생성
python -m scripts.merge_fruit_datasets --dst datasets/fruit10_merged

# 3. 학습 (imgsz=1280 사용 시 workers를 반드시 낮출 것 — host RAM OOM 위험)
python -m scripts.train_fruit_detector \
  --data datasets/fruit10_merged/dataset.yaml \
  --model yolov8s.pt \
  --epochs 30 --batch 8 --imgsz 1280 --workers 2 \
  --name fruit10_merged_v3
```

학습 후 `utils/detector.py`의 `DEFAULT_FRUIT_WEIGHTS`를 새 경로로 갱신하거나, 호출부에서 `weights=` 인자로 지정하세요.

현재 default(v2b, imgsz=1280) 결과 / 남은 개선 항목은 `to-do-work.md` 참조.

## 평가

```bash
# detector + SAM 파이프라인 IoU 평가 (greedy match by class)
python -m tests.eval_iou \
  --val-dir datasets/fruit10_merged/val \
  --max-images 300 \
  --weights runs/detect/runs/detect/fruit10_merged_v2b_imgsz1280/weights/best.pt \
  --out outputs/eval/eval_iou_v2b_full.json

# 단위 테스트
pytest tests/ -v
```

## 시스템 요구사항

- Ubuntu 22.04, Python 3.10+
- NVIDIA RTX 4060 8GB (또는 동급 CUDA GPU). imgsz=1280 학습은 호스트 RAM 16GB 이상 권장 (workers=2 기준)
- Node.js 18+ (프론트엔드 빌드)
- 디스크: SAM (ViT-B 0.4GB / ViT-H 2.4GB) + YOLO weights + 학습 데이터셋 (~10GB) + node_modules

## 문제 해결

| 증상 | 해결 |
|---|---|
| `CUDA out of memory` (추론) | `--sam_model vit_b`, `--max_size 768`, 또는 `imgsz` 낮추기 |
| `Linux OOM kill` (학습 중) | `--workers 2` 사용, imgsz=1280에서 default 8은 RAM 21GB 사용 → kill |
| `YOLO weights not found` | `DEFAULT_FRUIT_WEIGHTS` 경로 확인 또는 `--det_weights` 지정 |
| 프론트 `ECONNREFUSED` | 백엔드 8000 포트 구동 여부 확인 |
| CORS 오류 | `backend/config.py`의 `ALLOWED_ORIGINS`에 도메인 추가 |

## 참고

- [segment-anything (Meta)](https://github.com/facebookresearch/segment-anything)
- [Ultralytics YOLO](https://docs.ultralytics.com)
