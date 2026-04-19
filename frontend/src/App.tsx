import { useEffect } from "react";
import { useApp, type TabKey } from "./store";
import { createSession, health } from "./api/client";
import AutoLabelTab from "./tabs/AutoLabelTab";
import ReviewQueueTab from "./tabs/ReviewQueueTab";
import ExportTab from "./tabs/ExportTab";

const TABS: { key: TabKey; label: string }[] = [
  { key: "auto", label: "Auto Labeling" },
  { key: "review", label: "Review Queue" },
  { key: "export", label: "Export" },
];

export default function App() {
  const { sessionId, setSessionId, tab, setTab, classes, setClasses } = useApp();

  useEffect(() => {
    health().then((h) => setClasses(h.classes));
    if (!sessionId) {
      createSession().then((r) => setSessionId(r.session_id));
    }
  }, []);

  return (
    <div className="flex flex-col h-full">
      <header className="border-b border-slate-700 px-6 py-3 flex items-center gap-6">
        <h1 className="text-lg font-semibold">SAM Auto-Label Studio</h1>
        <span className="text-xs text-slate-400">
          session: {sessionId ?? "…"} · classes: {classes.length}
        </span>
        <nav className="ml-auto flex gap-2">
          {TABS.map((t) => (
            <button
              key={t.key}
              className={`px-3 py-1 rounded text-sm ${
                tab === t.key
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
              onClick={() => setTab(t.key)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="flex-1 overflow-hidden">
        {tab === "auto" && <AutoLabelTab />}
        {tab === "review" && <ReviewQueueTab />}
        {tab === "export" && <ExportTab />}
      </main>
    </div>
  );
}
