import { useEffect, useMemo, useState } from "react";
import {
  getLabels,
  imageFileUrl,
  listImages,
  refineMask,
  setLabels as postLabels,
  type ImageSummary,
  type LabelEntry,
} from "../api/client";
import { useApp } from "../store";
import MaskCanvas from "../components/MaskCanvas";

export default function ReviewQueueTab() {
  const { sessionId, classes, selectedImageId, setSelectedImageId } = useApp();
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [filter, setFilter] = useState<"all" | "uncertain" | "unreviewed">("all");
  const [labels, setLabels] = useState<LabelEntry[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [imgMeta, setImgMeta] = useState<{ w: number; h: number } | null>(null);
  const [dirty, setDirty] = useState(false);
  const [pointing, setPointing] = useState(false);
  const [boxing, setBoxing] = useState(false);
  const [pendingBox, setPendingBox] = useState<
    [number, number, number, number] | null
  >(null);
  const [pendingPoints, setPendingPoints] = useState<
    { imageX: number; imageY: number; pointLabel: 1 | 0 }[]
  >([]);
  const [refining, setRefining] = useState(false);

  const refreshList = async () => {
    if (!sessionId) return;
    setImages(await listImages(sessionId, filter === "uncertain"));
  };

  useEffect(() => {
    refreshList();
  }, [sessionId, filter]);

  const loadImage = async (imageId: string) => {
    if (!sessionId) return;
    const data = await getLabels(sessionId, imageId);
    setLabels(data.labels);
    setImgMeta({ w: data.width, h: data.height });
    setSelectedIdx(null);
    setDirty(false);
    setPendingPoints([]);
    setPointing(false);
    setPendingBox(null);
    setBoxing(false);
  };

  useEffect(() => {
    if (selectedImageId) loadImage(selectedImageId);
  }, [selectedImageId]);

  const filtered = useMemo(() => {
    switch (filter) {
      case "uncertain":
        return images.filter((i) => i.uncertain);
      case "unreviewed":
        return images.filter((i) => !i.reviewed);
      default:
        return images;
    }
  }, [images, filter]);

  const onClassChange = (clsId: number) => {
    if (selectedIdx === null) return;
    setLabels((prev) =>
      prev.map((l, i) =>
        i === selectedIdx
          ? { ...l, cls_id: clsId, cls_name: classes[clsId] ?? String(clsId) }
          : l
      )
    );
    setDirty(true);
  };

  const onDelete = () => {
    if (selectedIdx === null) return;
    setLabels((prev) => prev.filter((_, i) => i !== selectedIdx));
    setSelectedIdx(null);
    setDirty(true);
  };

  const onAddPoint = (ev: {
    imageX: number;
    imageY: number;
    normX: number;
    normY: number;
    pointLabel: 1 | 0;
  }) => {
    setPendingPoints((prev) => [
      ...prev,
      { imageX: ev.imageX, imageY: ev.imageY, pointLabel: ev.pointLabel },
    ]);
  };

  const applyRefineResult = (polygon: number[], samScore: number, bbox: number[] | null) => {
    setLabels((prev) => {
      const clsId = selectedIdx !== null ? prev[selectedIdx].cls_id : 0;
      const clsName = classes[clsId] ?? String(clsId);
      const newLabel: LabelEntry = {
        cls_id: clsId,
        cls_name: clsName,
        polygon,
        sam_score: samScore,
        det_conf: null,
        bbox,
      };
      if (selectedIdx !== null) {
        return prev.map((l, i) => (i === selectedIdx ? newLabel : l));
      }
      return [...prev, newLabel];
    });
    setDirty(true);
  };

  const onApplyRefine = async () => {
    if (!sessionId || !selectedImageId || pendingPoints.length === 0) return;
    setRefining(true);
    try {
      const res = await refineMask({
        session_id: sessionId,
        image_id: selectedImageId,
        points: pendingPoints.map((p) => [p.imageX, p.imageY]),
        point_labels: pendingPoints.map((p) => p.pointLabel),
      });
      applyRefineResult(res.polygon, res.sam_score, null);
      setPendingPoints([]);
      setPointing(false);
    } finally {
      setRefining(false);
    }
  };

  const onApplyBoxRefine = async () => {
    if (!sessionId || !selectedImageId || !pendingBox) return;
    setRefining(true);
    try {
      const res = await refineMask({
        session_id: sessionId,
        image_id: selectedImageId,
        bbox: pendingBox,
      });
      applyRefineResult(res.polygon, res.sam_score, pendingBox);
      setPendingBox(null);
      setBoxing(false);
    } finally {
      setRefining(false);
    }
  };

  const togglePointing = () => {
    if (pointing) {
      setPointing(false);
      setPendingPoints([]);
    } else {
      setPointing(true);
      setBoxing(false);
      setPendingBox(null);
    }
  };

  const toggleBoxing = () => {
    if (boxing) {
      setBoxing(false);
      setPendingBox(null);
    } else {
      setBoxing(true);
      setPointing(false);
      setPendingPoints([]);
    }
  };

  const onSave = async () => {
    if (!sessionId || !selectedImageId || !imgMeta) return;
    await postLabels(sessionId, selectedImageId, {
      session_id: sessionId,
      image_id: selectedImageId,
      width: imgMeta.w,
      height: imgMeta.h,
      labels,
    });
    setDirty(false);
    await refreshList();
  };

  return (
    <div className="flex h-full">
      <aside className="w-64 border-r border-slate-700 overflow-y-auto">
        <div className="p-3 flex gap-1 text-xs">
          {(["all", "uncertain", "unreviewed"] as const).map((f) => (
            <button
              key={f}
              className={`px-2 py-1 rounded ${
                filter === f ? "bg-indigo-600" : "bg-slate-800"
              }`}
              onClick={() => setFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>
        <ul>
          {filtered.map((im) => (
            <li key={im.image_id}>
              <button
                className={`w-full text-left px-3 py-2 text-sm border-b border-slate-800 hover:bg-slate-800 ${
                  im.image_id === selectedImageId ? "bg-slate-800" : ""
                }`}
                onClick={() => setSelectedImageId(im.image_id)}
              >
                <div className="truncate">{im.filename}</div>
                <div className="text-xs text-slate-400 flex gap-2">
                  <span>{im.n_labels} labels</span>
                  {im.uncertain && <span className="text-amber-400">uncertain</span>}
                  {im.reviewed && <span className="text-emerald-400">reviewed</span>}
                </div>
              </button>
            </li>
          ))}
          {filtered.length === 0 && (
            <li className="p-3 text-xs text-slate-500">해당 필터의 이미지가 없습니다.</li>
          )}
        </ul>
      </aside>

      <section className="flex-1 flex flex-col">
        {selectedImageId && imgMeta && sessionId ? (
          <MaskCanvas
            imageUrl={imageFileUrl(sessionId, selectedImageId)}
            width={imgMeta.w}
            height={imgMeta.h}
            labels={labels}
            selectedIndex={selectedIdx}
            onSelect={setSelectedIdx}
            classes={classes}
            pointingMode={pointing}
            onPoint={onAddPoint}
            boxMode={boxing}
            onBox={setPendingBox}
            pendingBox={pendingBox}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-500">
            좌측에서 이미지를 선택하세요.
          </div>
        )}
      </section>

      <aside className="w-72 border-l border-slate-700 p-3 overflow-y-auto flex flex-col gap-3">
        <div className="text-sm">
          선택된 마스크: {selectedIdx === null ? "없음" : `#${selectedIdx}`}
        </div>

        <div>
          <div className="text-xs text-slate-400 mb-1">클래스</div>
          <div className="grid grid-cols-2 gap-1">
            {classes.map((c, idx) => (
              <button
                key={c}
                disabled={selectedIdx === null}
                className="text-xs px-2 py-1 rounded bg-slate-800 hover:bg-indigo-600 disabled:opacity-40"
                onClick={() => onClassChange(idx)}
              >
                {c}
              </button>
            ))}
          </div>
        </div>

        <div className="border-t border-slate-700 pt-3">
          <div className="text-xs text-slate-400 mb-1">포인트 보정 (SAM)</div>
          <div className="flex gap-2 mb-2">
            <button
              className={`flex-1 text-xs py-2 rounded ${
                pointing ? "bg-amber-500" : "bg-slate-800 hover:bg-slate-700"
              }`}
              onClick={togglePointing}
            >
              {pointing ? "포인트 모드 종료" : "포인트 모드"}
            </button>
          </div>
          {pointing && (
            <div className="text-xs text-slate-400 mb-2">
              좌클릭 = 전경, 우클릭 = 배경 ({pendingPoints.length} pts)
            </div>
          )}
          <button
            className="w-full text-xs py-2 rounded bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700"
            disabled={pendingPoints.length === 0 || refining}
            onClick={onApplyRefine}
          >
            {refining ? "SAM 재분할 중…" : "포인트로 재분할"}
          </button>
        </div>

        <div className="border-t border-slate-700 pt-3">
          <div className="text-xs text-slate-400 mb-1">박스 보정 (SAM)</div>
          <div className="flex gap-2 mb-2">
            <button
              className={`flex-1 text-xs py-2 rounded ${
                boxing ? "bg-amber-500" : "bg-slate-800 hover:bg-slate-700"
              }`}
              onClick={toggleBoxing}
            >
              {boxing ? "박스 모드 종료" : "박스 모드"}
            </button>
          </div>
          {boxing && (
            <div className="text-xs text-slate-400 mb-2">
              {pendingBox
                ? `bbox: [${pendingBox.map((v) => v.toFixed(0)).join(", ")}]`
                : "이미지 위에서 드래그하여 박스를 그리세요."}
            </div>
          )}
          <button
            className="w-full text-xs py-2 rounded bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700"
            disabled={!pendingBox || refining}
            onClick={onApplyBoxRefine}
          >
            {refining ? "SAM 재분할 중…" : "박스로 재분할"}
          </button>
        </div>

        <div className="flex gap-2">
          <button
            className="flex-1 text-sm py-2 rounded bg-rose-600 hover:bg-rose-500 disabled:bg-slate-700"
            disabled={selectedIdx === null}
            onClick={onDelete}
          >
            삭제
          </button>
          <button
            className="flex-1 text-sm py-2 rounded bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700"
            disabled={!dirty}
            onClick={onSave}
          >
            저장
          </button>
        </div>

        <div className="border-t border-slate-700 pt-3 text-xs space-y-2">
          <div className="text-slate-400">마스크 리스트</div>
          {labels.map((l, i) => (
            <div
              key={i}
              className={`flex justify-between px-2 py-1 rounded cursor-pointer ${
                i === selectedIdx ? "bg-slate-800" : "hover:bg-slate-800/50"
              }`}
              onClick={() => setSelectedIdx(i)}
            >
              <span>
                #{i} {l.cls_name}
              </span>
              <span className="text-slate-500">
                {l.sam_score ? l.sam_score.toFixed(2) : "—"}
              </span>
            </div>
          ))}
          {labels.length === 0 && <div className="text-slate-500">마스크 없음</div>}
        </div>
      </aside>
    </div>
  );
}
