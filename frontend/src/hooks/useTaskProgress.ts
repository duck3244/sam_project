import { useEffect, useState } from "react";

export interface ProgressEvent {
  task_id: string;
  done: number;
  total: number;
  current_image?: string | null;
  uncertain_count: number;
  status: "running" | "completed" | "failed" | "pending";
  error?: string | null;
}

export function useTaskProgress(taskId: string | null) {
  const [event, setEvent] = useState<ProgressEvent | null>(null);

  useEffect(() => {
    if (!taskId) return;
    setEvent({ task_id: taskId, done: 0, total: 0, uncertain_count: 0, status: "pending" });
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws/progress/${taskId}`);
    ws.onmessage = (msg) => {
      try {
        setEvent(JSON.parse(msg.data));
      } catch {
        /* ignore */
      }
    };
    ws.onerror = () => {
      setEvent((p) =>
        p ? { ...p, status: "failed", error: "websocket error" } : p
      );
    };
    return () => ws.close();
  }, [taskId]);

  return event;
}
