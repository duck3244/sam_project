interface Props {
  done: number;
  total: number;
  label?: string;
}

export default function ProgressBar({ done, total, label }: Props) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-slate-400 mb-1">
        <span>{label ?? "progress"}</span>
        <span>
          {done} / {total} ({pct}%)
        </span>
      </div>
      <div className="w-full h-2 bg-slate-800 rounded overflow-hidden">
        <div
          className="h-full bg-indigo-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
