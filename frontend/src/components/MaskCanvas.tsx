import { useEffect, useRef, useState } from "react";
import type { LabelEntry } from "../api/client";

interface Props {
  imageUrl: string;
  width: number;
  height: number;
  labels: LabelEntry[];
  selectedIndex: number | null;
  onSelect: (index: number | null) => void;
  onPoint?: (ev: {
    imageX: number;
    imageY: number;
    normX: number;
    normY: number;
    pointLabel: 1 | 0;
  }) => void;
  pointingMode?: boolean;
  boxMode?: boolean;
  onBox?: (bbox: [number, number, number, number]) => void;
  pendingBox?: [number, number, number, number] | null;
  classes: string[];
}

const CLASS_COLORS = [
  "#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#84cc16", "#06b6d4",
];

export default function MaskCanvas({
  imageUrl,
  width,
  height,
  labels,
  selectedIndex,
  onSelect,
  onPoint,
  pointingMode,
  boxMode,
  onBox,
  pendingBox,
  classes,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [containerSize, setContainerSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<
    | { startX: number; startY: number; curX: number; curY: number }
    | null
  >(null);

  useEffect(() => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => setImg(image);
    image.src = imageUrl;
    return () => {
      image.onload = null;
    };
  }, [imageUrl]);

  useEffect(() => {
    if (!wrapperRef.current) return;
    const ro = new ResizeObserver(() => {
      const el = wrapperRef.current!;
      setContainerSize({ w: el.clientWidth, h: el.clientHeight });
    });
    ro.observe(wrapperRef.current);
    return () => ro.disconnect();
  }, []);

  const scale = (() => {
    if (!img || containerSize.w === 0 || containerSize.h === 0) return 1;
    return Math.min(containerSize.w / width, containerSize.h / height);
  })();
  const displayW = width * scale;
  const displayH = height * scale;

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !img) return;
    cv.width = displayW;
    cv.height = displayH;
    const ctx = cv.getContext("2d")!;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(img, 0, 0, displayW, displayH);

    labels.forEach((lbl, idx) => {
      if (!lbl.polygon || lbl.polygon.length < 6) return;
      const color = CLASS_COLORS[lbl.cls_id % CLASS_COLORS.length];
      const selected = idx === selectedIndex;
      ctx.beginPath();
      for (let i = 0; i < lbl.polygon.length; i += 2) {
        const px = lbl.polygon[i] * displayW;
        const py = lbl.polygon[i + 1] * displayH;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = color + (selected ? "66" : "33");
      ctx.fill();
      ctx.lineWidth = selected ? 3 : 1.5;
      ctx.strokeStyle = color;
      ctx.stroke();

      const firstX = lbl.polygon[0] * displayW;
      const firstY = lbl.polygon[1] * displayH;
      ctx.font = "12px sans-serif";
      ctx.fillStyle = color;
      ctx.fillText(
        `${classes[lbl.cls_id] ?? lbl.cls_id}`,
        firstX + 4,
        Math.max(12, firstY - 4)
      );
    });

    // Live drag rectangle
    if (drag) {
      const x = Math.min(drag.startX, drag.curX);
      const y = Math.min(drag.startY, drag.curY);
      const w = Math.abs(drag.curX - drag.startX);
      const h = Math.abs(drag.curY - drag.startY);
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = "#fbbf24";
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    } else if (pendingBox) {
      const [bx1, by1, bx2, by2] = pendingBox;
      const x = (bx1 / width) * displayW;
      const y = (by1 / height) * displayH;
      const w = ((bx2 - bx1) / width) * displayW;
      const h = ((by2 - by1) / height) * displayH;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = "#fbbf24";
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }
  }, [img, labels, selectedIndex, displayW, displayH, classes, drag, pendingBox, width, height]);

  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { cx: e.clientX - rect.left, cy: e.clientY - rect.top };
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!img || boxMode) return;
    const { cx, cy } = getCanvasCoords(e);
    const normX = cx / displayW;
    const normY = cy / displayH;
    const imageX = normX * width;
    const imageY = normY * height;

    if (pointingMode && onPoint) {
      e.preventDefault();
      const pointLabel: 1 | 0 = e.button === 2 ? 0 : 1;
      onPoint({ imageX, imageY, normX, normY, pointLabel });
      return;
    }

    for (let i = labels.length - 1; i >= 0; i--) {
      const poly = labels[i].polygon;
      if (!poly || poly.length < 6) continue;
      if (pointInPolygon(normX, normY, poly)) {
        onSelect(i);
        return;
      }
    }
    onSelect(null);
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!img || !boxMode || e.button !== 0) return;
    e.preventDefault();
    const { cx, cy } = getCanvasCoords(e);
    setDrag({ startX: cx, startY: cy, curX: cx, curY: cy });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drag) return;
    const { cx, cy } = getCanvasCoords(e);
    setDrag({ ...drag, curX: cx, curY: cy });
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drag || !onBox) {
      setDrag(null);
      return;
    }
    const { cx, cy } = getCanvasCoords(e);
    const x1 = Math.min(drag.startX, cx);
    const y1 = Math.min(drag.startY, cy);
    const x2 = Math.max(drag.startX, cx);
    const y2 = Math.max(drag.startY, cy);
    setDrag(null);
    if (x2 - x1 < 4 || y2 - y1 < 4) return;
    const sx = width / displayW;
    const sy = height / displayH;
    onBox([x1 * sx, y1 * sy, x2 * sx, y2 * sy]);
  };

  const cursor = boxMode ? "crosshair" : pointingMode ? "crosshair" : "pointer";

  return (
    <div
      ref={wrapperRef}
      className="w-full h-full bg-black flex items-center justify-center overflow-hidden"
    >
      <canvas
        ref={canvasRef}
        style={{ cursor }}
        onClick={handleClick}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => setDrag(null)}
        onContextMenu={(e) => {
          e.preventDefault();
          handleClick(e);
        }}
      />
    </div>
  );
}

function pointInPolygon(x: number, y: number, poly: number[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 2; i < poly.length; j = i, i += 2) {
    const xi = poly[i];
    const yi = poly[i + 1];
    const xj = poly[j];
    const yj = poly[j + 1];
    const intersect =
      yi > y !== yj > y &&
      x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}
