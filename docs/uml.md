# UML Diagrams

SAM Auto-Label Studio의 클래스/시퀀스/컴포넌트 다이어그램. Mermaid 문법 사용 — GitHub/PyCharm에서 직접 렌더링.

> 시스템 전반 설명은 [`architecture.md`](./architecture.md) 참고.

---

## 1. Component Diagram (System-level)

```mermaid
flowchart LR
    subgraph FE["Frontend (React + Vite)"]
        App["App.tsx"]
        Auto["AutoLabelTab"]
        Review["ReviewQueueTab"]
        Export["ExportTab"]
        Canvas["MaskCanvas"]
        Hook["useTaskProgress"]
        Store[("Zustand store")]
        Client["api/client.ts (Axios)"]
    end

    subgraph BE["Backend (FastAPI)"]
        Main["main.py"]
        SessionR["routers/session"]
        PipelineR["routers/pipeline (+WS)"]
        ExportR["routers/export"]
        SS["SessionStore (singleton)"]
        DS["detector_service"]
        SamS["sam_service"]
    end

    subgraph ML["ML core (utils/)"]
        FD["FruitDetector"]
        SP["SAMPredictor"]
        FLP["FruitLabelPipeline"]
        Exp["export.mask_to_yolo_polygon"]
        Unc["uncertainty"]
    end

    subgraph FS["Filesystem"]
        Sessions[("outputs/sessions/")]
        Models[("models/ SAM ckpt")]
        Runs[("runs/ YOLO weights")]
    end

    Auto -->|HTTP| Client
    Review -->|HTTP| Client
    Export -->|HTTP| Client
    Hook -->|WS| PipelineR
    Client -->|REST| SessionR
    Client -->|REST| PipelineR
    Client -->|REST| ExportR

    SessionR --> SS
    PipelineR --> SS
    PipelineR --> DS
    PipelineR --> SamS
    PipelineR --> FLP
    ExportR --> SS
    ExportR --> Exp

    DS --> FD
    SamS --> SP
    FLP --> FD
    FLP --> SP
    FLP --> Exp
    FLP --> Unc

    FD -.loads.-> Runs
    SP -.loads.-> Models
    SS -.persists imgs.-> Sessions
```

---

## 2. Class Diagram — Backend

```mermaid
classDiagram
    direction LR

    class Session {
        +str session_id
        +Path root
        +dict~str,ImageRecord~ images
        +datetime created_at
        +Path images_dir
        +Path labels_dir
    }

    class ImageRecord {
        +str image_id
        +Path path
        +int width
        +int height
        +list~dict~ labels
        +bool uncertain
        +bool reviewed
        +float priority
    }

    class Task {
        +str task_id
        +str session_id
        +int total
        +int done
        +str current_image
        +int uncertain_count
        +str status
        +str error
    }

    class SessionStore {
        -dict _sessions
        -dict _tasks
        -dict _subscribers
        -Lock _lock
        +create_session() Session
        +require_session(sid) Session
        +add_image(sid, path, w, h) ImageRecord
        +list_images(sid, sort) list~ImageRecord~
        +set_labels(sid, iid, labels)
        +mark_uncertain(sid, iid, flag)
        +create_task(sid, total) Task
        +subscribe(tid) Queue
        +unsubscribe(tid, q)
        +publish(tid, payload)
    }

    class FruitDetector {
        -model: YOLO
        +str device
        +float conf
        +float iou
        +int imgsz
        +detect(image) list~dict~
        +detect_batch(images) list~list~
    }

    class SAMPredictor {
        -predictor
        -mask_generator
        +str device
        +set_image(image)
        +predict_with_box(box) tuple
        +predict_with_points(coords, labels) tuple
        +predict_with_points_and_box(...) tuple
        +get_best_mask(masks, scores) tuple
        +generate_masks(image) list
    }

    class FruitLabelPipeline {
        +SAMPredictor sam
        +FruitDetector detector
        +tuple class_names
        +float epsilon
        +int max_image_size
        +bool multi_contour
        +float min_area_ratio
        +process(image_path, out_dir) LabelResult
        +process_batch(paths, out_dir, cb) list
    }

    class LabelResult {
        +Path image_path
        +Path label_path
        +int n_labels
        +int n_detections
        +list per_label
        +str error
        +bool ok
    }

    class UncertaintyThresholds {
        +float min_score
        +float max_variance
        +float min_det_conf
    }

    Session "1" *-- "*" ImageRecord : contains
    SessionStore "1" o-- "*" Session : tracks
    SessionStore "1" o-- "*" Task : tracks
    FruitLabelPipeline --> FruitDetector : uses
    FruitLabelPipeline --> SAMPredictor : uses
    FruitLabelPipeline ..> LabelResult : produces
```

---

## 3. Class Diagram — Frontend (Zustand + Tabs)

```mermaid
classDiagram
    direction TB

    class AppStore {
        <<zustand>>
        +string sessionId
        +TabKey tab
        +string[] classes
        +SessionStats stats
        +string selectedImageId
        +setSessionId(id)
        +setTab(t)
        +setClasses(c)
        +setStats(s)
        +setSelectedImageId(id)
    }

    class App {
        +useEffect()
        +render()
    }

    class AutoLabelTab {
        +ImageSummary[] images
        +string samModel
        +number detConf
        +bool multiContour
        +string taskId
        +bool busy
        +useTaskProgress(taskId)
        +onUpload()
        +onStart()
    }

    class ReviewQueueTab {
        +ImageSummary[] images
        +Filter filter
        +LabelEntry[] labels
        +number selectedIdx
        +bool dirty
        +bool pointing
        +bool boxing
        +onLoadImage()
        +onApplyRefine()
        +onSave()
    }

    class ExportTab {
        +SessionStats stats
        +useEffect()
    }

    class MaskCanvas {
        +string imageUrl
        +LabelEntry[] labels
        +number selectedIndex
        +onSelect(i)
        +onPoint(ev)
        +onBox(bbox)
    }

    class useTaskProgress {
        <<hook>>
        +string taskId
        +ProgressEvent event
        +ws connection
    }

    class ApiClient {
        <<axios module>>
        +health()
        +createSession()
        +uploadImages(sid, files)
        +listImages(sid, sort)
        +getLabels(sid, iid)
        +setLabels(sid, iid, body)
        +runPipeline(sid, req)
        +getStats(sid)
        +refineMask(body)
        +imageFileUrl(sid, iid)
        +exportZipUrl(sid)
    }

    App --> AppStore : reads/writes
    App --> AutoLabelTab
    App --> ReviewQueueTab
    App --> ExportTab
    AutoLabelTab --> AppStore
    AutoLabelTab --> ApiClient
    AutoLabelTab --> useTaskProgress
    ReviewQueueTab --> AppStore
    ReviewQueueTab --> ApiClient
    ReviewQueueTab --> MaskCanvas
    ExportTab --> AppStore
    ExportTab --> ApiClient
```

---

## 4. Sequence Diagram — Auto-Labeling Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant FE as AutoLabelTab
    participant API as Axios client
    participant R as routers/pipeline
    participant SS as SessionStore
    participant DS as detector_service
    participant SamS as sam_service
    participant FLP as FruitLabelPipeline
    participant WS as WebSocket /progress

    U->>FE: drop files + click Start
    FE->>API: uploadImages(sid, files)
    API->>R: POST /session/{sid}/images
    R->>SS: add_image(sid, path, w, h)
    SS-->>R: ImageRecord[]
    R-->>API: image_ids
    FE->>API: runPipeline(sid, params)
    API->>R: POST /pipeline/{sid}/run
    R->>SS: create_task(sid, total)
    SS-->>R: Task(task_id)
    R-->>API: {task_id}
    API-->>FE: task_id
    FE->>WS: connect /ws/progress/{task_id}

    par background async
        R->>DS: get_detector(conf)
        DS-->>R: FruitDetector (singleton)
        R->>SamS: get_sam(model_type)
        SamS-->>R: SAMPredictor (cached)
        loop for each image
            R->>FLP: process(image_path, labels_dir)
            FLP->>FLP: detect → set_image → predict_with_box → mask_to_yolo_polygon
            FLP-->>R: LabelResult(per_label[])
            R->>SS: set_labels + mark_uncertain + priority
            R->>SS: publish(task_id, ProgressEvent)
            SS-->>WS: ProgressEvent JSON
            WS-->>FE: onmessage
        end
        R->>SS: publish(... status="completed")
    end
    FE->>FE: setEvent → refresh image grid
```

---

## 5. Sequence Diagram — Refine Mask

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant Canvas as MaskCanvas
    participant FE as ReviewQueueTab
    participant API as Axios client
    participant R as routers/pipeline
    participant SamS as sam_service
    participant SP as SAMPredictor

    U->>Canvas: click(point) or drag(box)
    Canvas->>FE: onPoint / onBox
    FE->>API: refineMask({sid, iid, points|bbox, ...})
    API->>R: POST /refine
    R->>R: load_image + resize_image
    R->>SamS: get_sam(DEFAULT_SAM_MODEL)
    SamS-->>R: SAMPredictor
    R->>SP: set_image(resized)
    alt points provided
        R->>SP: predict_with_points(coords, labels)
    else bbox provided
        R->>SP: predict_with_box(box)
    end
    SP-->>R: (masks, scores, logits)
    R->>SP: get_best_mask
    SP-->>R: best_mask, best_score
    R->>R: mask_to_yolo_polygon(best_mask, w, h, ε)
    R-->>API: {polygon, sam_score}
    API-->>FE: refine result
    FE->>FE: applyRefineResult → labels[i].polygon = polygon, setDirty(true)
    Note over FE: 사용자가 Save 누를 때까지<br/>store에는 반영 안 됨
```

---

## 6. Sequence Diagram — Save & Export

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant FE as ReviewQueueTab
    participant Exp as ExportTab
    participant API as Axios client
    participant R1 as routers/session
    participant R2 as routers/export
    participant SS as SessionStore
    participant Pkg as zipfile

    U->>FE: Save
    FE->>API: setLabels(sid, iid, {labels})
    API->>R1: POST /session/{sid}/images/{iid}/labels
    R1->>SS: set_labels(sid, iid, labels) (in-memory)
    R1-->>API: ImageLabels (echo)

    U->>Exp: 탭 전환 → mount
    Exp->>API: getStats(sid)
    API->>R1: GET /session/{sid}/stats
    R1->>SS: list_images / count
    R1-->>Exp: SessionStats

    U->>Exp: Download click
    Exp->>API: GET /export/{sid}?format=yolo-seg
    API->>R2: route
    R2->>SS: list_images(sid)
    SS-->>R2: ImageRecord[]
    loop each ImageRecord
        R2->>Pkg: write labels/{stem}.txt
    end
    R2->>Pkg: write dataset.yaml
    Pkg-->>R2: zip buffer
    R2-->>API: StreamingResponse(zip)
    API-->>U: download yolo-seg.zip
```

---

## 7. Sequence Diagram — Detector Retraining (offline)

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Developer
    participant Hh as remap_henningheyen
    participant Dn as remap_deepnir_mango
    participant Mg as merge_fruit_datasets
    participant Tr as train_fruit_detector
    participant U as ultralytics.YOLO
    participant Det as utils/detector.py

    Dev->>Hh: --src LVIS_Fruits --dst datasets/henningheyen_fruit10
    Hh-->>Dev: remapped (10-class taxonomy)
    Dev->>Dn: --src deepNIR --dst datasets/deepNIR_mango --mango-cls 6
    Dn-->>Dev: mango-only YOLO
    Dev->>Mg: --hh ... --deepnir ... --dst datasets/fruit10_merged
    Mg->>Mg: 3 sources merge + banana cap + dataset.yaml 작성
    Mg-->>Dev: datasets/fruit10_merged/{train,val,test}/...
    Dev->>Tr: --data ... --imgsz 1280 --batch 8 --workers 2 --epochs 30
    Tr->>U: model.train(...)
    U-->>Tr: best.pt
    Tr-->>Dev: runs/detect/.../weights/best.pt
    Dev->>Det: edit DEFAULT_FRUIT_WEIGHTS
    Note over Det: 다음 inference부터 새 detector 사용
```

---

## 8. Data Model — In-memory Entities

```mermaid
erDiagram
    SESSION ||--o{ IMAGE_RECORD : contains
    SESSION ||--o{ TASK : owns

    SESSION {
        string session_id PK
        path   root
        datetime created_at
    }

    IMAGE_RECORD {
        string image_id PK
        string session_id FK
        path   path
        int    width
        int    height
        list   labels
        bool   uncertain
        bool   reviewed
        float  priority
    }

    TASK {
        string task_id PK
        string session_id FK
        int    total
        int    done
        string current_image
        int    uncertain_count
        string status
        string error
    }

    LABEL_ENTRY {
        int    cls_id
        string cls_name
        list   polygon
        float  sam_score
        float  det_conf
        list   bbox
    }

    IMAGE_RECORD ||--o{ LABEL_ENTRY : has
```

---

## 9. State Machine — Task Lifecycle

```mermaid
stateDiagram-v2
    [*] --> running : create_task(sid, total)
    running --> running : publish ProgressEvent (per image)
    running --> completed : all images processed
    running --> failed : exception
    completed --> [*] : WS clients disconnect
    failed --> [*] : WS clients disconnect
```

---

## 10. WebSocket Channel Lifecycle

```mermaid
sequenceDiagram
    participant FE as useTaskProgress
    participant WS as /ws/progress/{task_id}
    participant SS as SessionStore

    FE->>WS: connect
    WS->>SS: subscribe(task_id) → Queue
    WS->>FE: initial Task snapshot
    loop until status in [completed, failed]
        SS->>WS: publish(payload) → queue.put_nowait
        WS->>FE: send_json(payload)
    end
    FE->>WS: close()
    WS->>SS: unsubscribe(task_id, queue)
```
