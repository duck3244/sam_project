import { useEffect, useRef, useState } from "react";
import {
  listImages,
  runPipeline,
  uploadImages,
  imageFileUrl,
  type ImageSummary,
} from "../api/client";
import { useApp } from "../store";
import { useTaskProgress } from "../hooks/useTaskProgress";
import ProgressBar from "../components/ProgressBar";

type SamModel = "vit_b" | "vit_h";

export default function AutoLabelTab() {
  const { sessionId, setSelectedImageId, setTab } = useApp();
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [samModel, setSamModel] = useState<SamModel>("vit_b");
  const [detConf, setDetConf] = useState(0.15);
  const [multiContour, setMultiContour] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const progress = useTaskProgress(taskId);

  const refresh = async () => {
    if (!sessionId) return;
    setImages(await listImages(sessionId));
  };

  useEffect(() => {
    refresh();
  }, [sessionId]);

  useEffect(() => {
    if (progress?.status === "completed" || progress?.status === "failed") {
      setBusy(false);
      refresh();
    } else if (progress?.status === "running") {
      // optimistic refresh on every few ticks
      if (progress.done % 5 === 0) refresh();
    }
  }, [progress?.status, progress?.done]);

  const onFiles = async (files: FileList | null) => {
    if (!files || !sessionId) return;
    setUploading(true);
    try {
      await uploadImages(sessionId, Array.from(files));
      await refresh();
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const onRun = async () => {
    if (!sessionId || busy) return;
    setBusy(true);
    const r = await runPipeline(sessionId, {
      sam_model: samModel,
      det_conf: detConf,
      multi_contour: multiContour,
    });
    setTaskId(r.task_id);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    onFiles(e.dataTransfer.files);
  };

  const openReview = (id: string) => {
    setSelectedImageId(id);
    setTab("review");
  };

  const totalImages = images.length;
  const labeledCount = images.filter((i) => i.n_labels > 0).length;

  return (
    <div className="h-full flex flex-col p-6 gap-4">
      <section className="flex flex-wrap items-end gap-4">
        <label
          className="flex-1 min-w-[280px] border-2 border-dashed border-slate-700 rounded p-6 text-center text-slate-400 cursor-pointer hover:border-indigo-500"
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
        >
          <input
            type="file"
            multiple
            accept="image/*"
            hidden
            ref={fileInputRef}
            onChange={(e) => onFiles(e.target.files)}
          />
          <div onClick={() => fileInputRef.current?.click()}>
            {uploading ? "업로드 중…" : "이미지를 드래그하거나 클릭해서 선택"}
          </div>
          <div className="text-xs mt-1">
            현재 {totalImages}장 업로드됨 · {labeledCount}장 라벨링 완료
          </div>
        </label>

        <div className="flex flex-col gap-1 text-sm">
          <label>SAM 모델</label>
          <select
            className="bg-slate-800 rounded px-2 py-1"
            value={samModel}
            onChange={(e) => setSamModel(e.target.value as SamModel)}
            disabled={busy}
          >
            <option value="vit_b">ViT-B (빠름)</option>
            <option value="vit_h">ViT-H (정확)</option>
          </select>
        </div>

        <div className="flex flex-col gap-1 text-sm">
          <label>Detector conf: {detConf.toFixed(2)}</label>
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.05}
            value={detConf}
            onChange={(e) => setDetConf(parseFloat(e.target.value))}
            disabled={busy}
          />
        </div>

        <label className="flex items-center gap-2 text-sm select-none">
          <input
            type="checkbox"
            checked={multiContour}
            onChange={(e) => setMultiContour(e.target.checked)}
            disabled={busy}
          />
          멀티 컨투어 (분리된 블롭 보존)
        </label>

        <button
          className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 rounded px-4 py-2 text-sm"
          onClick={onRun}
          disabled={busy || totalImages === 0}
        >
          {busy ? "라벨링 중…" : "라벨링 시작"}
        </button>
      </section>

      {progress && (
        <section className="bg-slate-800/50 rounded p-3">
          <ProgressBar
            done={progress.done}
            total={progress.total}
            label={
              progress.status === "completed"
                ? "완료"
                : progress.status === "failed"
                ? `실패: ${progress.error ?? "unknown"}`
                : `처리 중… ${progress.current_image ?? ""}`
            }
          />
          {progress.uncertain_count > 0 && (
            <div className="text-xs text-amber-400 mt-1">
              불확실 샘플 {progress.uncertain_count}개 → Review Queue 확인
            </div>
          )}
        </section>
      )}

      <section className="flex-1 overflow-auto">
        {images.length === 0 ? (
          <div className="text-slate-500 text-sm">업로드된 이미지가 없습니다.</div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {images.map((im) => (
              <button
                key={im.image_id}
                className="group relative rounded overflow-hidden bg-slate-800 aspect-square border border-slate-700 hover:border-indigo-500"
                onClick={() => openReview(im.image_id)}
                title={im.filename}
              >
                {sessionId && (
                  <img
                    src={imageFileUrl(sessionId, im.image_id)}
                    alt={im.filename}
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                )}
                <div className="absolute bottom-0 inset-x-0 bg-black/60 text-xs px-2 py-1 flex justify-between">
                  <span>{im.n_labels} labels</span>
                  {im.uncertain && <span className="text-amber-400">!</span>}
                  {im.reviewed && <span className="text-emerald-400">✓</span>}
                </div>
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
