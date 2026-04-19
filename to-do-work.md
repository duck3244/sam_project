# Fruit Detector — 남은 작업 메모

작성: 2026-04-19 (v2b 학습 종료 시점)
현재 default 가중치: `runs/detect/runs/detect/fruit10_merged_v2b_imgsz1280/weights/best.pt`
현재 dataset: `datasets/fruit10_merged/` (10 classes)

## 현재 상태 (v2b 결과)

| Class | mAP50 (v2b) | 비고 |
|---|---|---|
| mango | 0.971 | 최고 (close-up deepNIR 효과) |
| orange | 0.624 | |
| apple | 0.576 | |
| pineapple | 0.518 | |
| strawberry | 0.473 | |
| banana | 0.461 | |
| watermelon | **0.430** | v1 0.30 → 큰 개선 (imgsz=1280 효과) |
| grape | 0.290 | flat |
| cherry | **0.036** | **v1 0.20 → 회귀** |
| peach | 0.022 | 사실상 미학습 (v1 0.02에서 변화 없음) |
| **all** | **0.440 / 0.311 (mAP50/mAP50-95)** | v1 0.43 / 0.30 대비 소폭 ↑ |

**confusion matrix 진단**: cherry는 70%, peach는 63%가 background로 빠짐 (= 클래스 혼동이 아니라 완전 미검출). 두 클래스 모두 같은 처방 (close-up 데이터 / 증강) 필요.

---

## TODO 1 — Cherry 회귀 복구 (데이터 준비됨, 학습만 보류)

**Why:** v2b에서 cherry mAP50이 0.20 → 0.036으로 회귀. 원인은 클래스 혼동이 아닌 70% background 미검출. 기존 LVIS/henningheyen cherry 데이터(903 box, wide-shot)가 imgsz=1280 fine-tune 과정에서 다른 우세 클래스에 밀려난 것으로 추정.

**준비 완료:**
- `datasets/deepNIR_kaggle/cherry/` — 154장 / 3,095 box, **close-up 도메인** (박스 ~13-20%, LVIS tiny와 다름)
- 출처 cls_id=0 → 우리 cherry=9 로 remap 필요
- mango와 동일 출처(Inkyu Sa / deepNIR)이므로 검증된 패턴

**To do:**
1. `scripts/remap_deepnir_cherry.py` 작성 (`remap_deepnir_mango.py`의 cherry 변형 — `OUR_*_CLS = 9`, src cls = 0)
2. `python -m scripts.remap_deepnir_cherry --src datasets/deepNIR_kaggle/cherry --dst datasets/deepNIR_cherry --src-cls 0` 실행
3. `merge_fruit_datasets.py`에 `--deepnir-cherry` 인자 추가 (또는 `--extra` 일반화)
4. merge 재실행 → `datasets/fruit10_merged_v3/`
5. fine-tune from v2b best.pt, `imgsz=1280 batch=8 workers=2 epochs=20-30 patience=10`. 호스트 RAM 21GB 한계 주의 (workers=2 필수)

**예상:** cherry mAP50 0.036 → 0.5+ (mango가 같은 데이터 패턴에서 0.97 달성). 다른 클래스 영향 추적 필요.

---

## TODO 2 — Peach 데이터 확보 후 재학습

**Why:** peach mAP50 0.02. 원인은 동일 — 기존 LVIS/COCO peach가 orchard 광각이라 박스 74%가 <0.5% area. 도메인이 잘못됨. imgsz=1280만으로는 해결 안 됨 (v2b에서도 0.022로 변화 없음).

**후보 데이터 (우선순위):**
1. **YOLO-Peach (MDPI Agronomy 2024, 14(8), 1628)** — 2,270장 close-up peach seedling, YOLO 포맷.
   - 다운로드 링크 미공개, corresponding author Yi Shi (Henan University of Science and Technology) 이메일 요청 필요
   - https://www.mdpi.com/2073-4395/14/8/1628 — Data Availability 섹션 참조
2. **EMA-YOLO immature yellow peach (Sensors 2024, 24(12), 3783)** — close-up homemade dataset, 크기 미공개. 백업.
3. **Roboflow Universe 추가 발굴** — 직접 검색 시 403, 별도 API key 또는 수동 브라우징 필요
4. **기존 LVIS peach 필터링 (no new data)** — `datasets/LVIS_Fruits_And_Vegetables`에서 peach 박스 중 area ≥1% 인 것만 추출. 양은 줄지만 도메인이 깨끗해질 가능성

**To do:**
1. 데이터 확보 (1 또는 2 또는 3 또는 4)
2. remap → cherry와 같은 패턴으로 단일 클래스 추출, our cls=8
3. merge 재실행 (cherry까지 포함된 v3 위에 추가하면 v4)
4. fine-tune (cherry 보강과 묶어서 1회 학습으로 처리하면 ~7h 절감)

---

## TODO 3 — FruitDetector 추론 imgsz 정렬

**Why:** `utils/detector.py`의 `FruitDetector.__init__(imgsz=640)`이 default. 학습은 1280에서 했는데 추론은 640으로 하면 small-object 학습 효과를 100% 활용 못함 (특히 watermelon/grape/strawberry).

**To do:**
- `imgsz` default를 1280으로 올리거나, 호출부(label pipeline / batch_process / eval_iou)에서 명시 전달
- Trade-off: 추론 ~4× 느려짐. 배치 처리는 OK, real-time UI 사용처는 주의
- `tests/eval_iou.py` 재실행 시 `--max-image-size 1280` 옵션 함께 검토

---

## TODO 4 — SAM-pipeline IoU 평가 표본 확장

**Why:** 현재 `outputs/eval/eval_iou_v2b.json`은 50장만 사용. 그 50장에 peach/cherry/watermelon GT가 0이라 watermelon 개선(+0.13)이 보이지 않고 watermelon FP만 부각됨 (P drop).

**To do:**
- `python -m tests.eval_iou --val-dir datasets/fruit10_merged/val --max-images 300+ --weights ... --out outputs/eval/eval_iou_v2b_full.json`
- 클래스별 P/R가 의미 있게 나오려면 클래스당 GT ≥ 30 정도 확보 필요. v2 retrain 전에 baseline 수치로 박아두면 비교 깔끔

---

## TODO 5 — 메모리 및 문서 정리

- `MEMORY.md`에 `detector_retrain_v2b.md` 등재 완료. v1 메모리는 historical로 유지
- `README.md`, `QUICKSTART.md` — 학습 weight 경로/사용법 언급 있다면 v2b로 업데이트 필요 여부 검토
- `outputs/eval/` 결과 정리 (v1, v2, v2b 비교 표 만들면 향후 의사결정에 유용)

---

## 참고 — 학습 시 주의사항 (v2 OOM 경험)

- imgsz=1280, batch=8, **workers=8 → 호스트 RAM 21GB 사용 → Linux OOM kill** (시스템 31GB)
- `workers=2`로 떨어뜨리면 shmem ~4GB로 안전. v2b는 이 설정으로 7h 완주
- GPU memory는 batch=8 / imgsz=1280 에서 ~7GB / 8GB 사용. TaskAlignedAssigner CUDA OOM은 자동 CPU fallback 경고만 발생, 학습 차단 안 함
- `scripts/train_fruit_detector.py`의 `--workers` default 8 → imgsz≥1280 사용 시 명시적으로 2-4로 낮출 것

## 참고 — 디스크 위치 요약

| 항목 | 경로 |
|---|---|
| 현재 default 가중치 | `runs/detect/runs/detect/fruit10_merged_v2b_imgsz1280/weights/best.pt` |
| v1 historical | `runs/detect/runs/detect/fruit10_merged_v1/weights/best.pt` |
| 통합 dataset | `datasets/fruit10_merged/` |
| cherry 추가 소스 | `datasets/deepNIR_kaggle/cherry/` |
| mango 소스 | `datasets/deepNIR_mango/` (이미 merged) |
| 평가 결과 | `outputs/eval/eval_iou_v2b.json` |
| 학습 로그 | `runs/train_v1.log`, `runs/train_v2b.log` |
