import { useEffect, useState } from "react";
import { exportZipUrl, getStats, type SessionStats } from "../api/client";
import { useApp } from "../store";

export default function ExportTab() {
  const { sessionId } = useApp();
  const [stats, setStats] = useState<SessionStats | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    getStats(sessionId).then(setStats);
  }, [sessionId]);

  const maxCount = Math.max(
    1,
    ...Object.values(stats?.class_distribution ?? {})
  );

  return (
    <div className="p-6 flex flex-col gap-6">
      <section className="flex gap-6">
        <StatCard label="이미지" value={stats?.image_count ?? 0} />
        <StatCard label="라벨링 완료" value={stats?.labeled_count ?? 0} />
        <StatCard label="총 마스크" value={stats?.total_labels ?? 0} />
        <StatCard
          label="불확실 샘플"
          value={stats?.uncertain_queue ?? 0}
          color="text-amber-400"
        />
      </section>

      <section>
        <h2 className="text-sm text-slate-400 mb-2">클래스 분포</h2>
        <div className="space-y-1">
          {Object.entries(stats?.class_distribution ?? {}).map(([name, n]) => (
            <div key={name} className="flex items-center gap-2 text-xs">
              <div className="w-20 text-slate-300">{name}</div>
              <div className="flex-1 h-3 bg-slate-800 rounded overflow-hidden">
                <div
                  className="h-full bg-indigo-500"
                  style={{ width: `${(n / maxCount) * 100}%` }}
                />
              </div>
              <div className="w-10 text-right text-slate-400">{n}</div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <a
          href={sessionId ? exportZipUrl(sessionId) : "#"}
          className={`inline-block px-5 py-2 rounded text-sm ${
            sessionId
              ? "bg-emerald-600 hover:bg-emerald-500 text-white"
              : "bg-slate-700 text-slate-400 pointer-events-none"
          }`}
          download
        >
          YOLO-seg zip 다운로드
        </a>
        <div className="text-xs text-slate-400 mt-2">
          labels/*.txt + dataset.yaml 포함
        </div>
      </section>
    </div>
  );
}

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color?: string;
}) {
  return (
    <div className="bg-slate-800 rounded px-4 py-3">
      <div className="text-xs text-slate-400">{label}</div>
      <div className={`text-2xl font-semibold ${color ?? ""}`}>{value}</div>
    </div>
  );
}
