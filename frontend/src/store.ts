import { create } from "zustand";
import type { SessionStats } from "./api/client";

export type TabKey = "auto" | "review" | "export";

interface AppState {
  sessionId: string | null;
  setSessionId: (id: string | null) => void;
  tab: TabKey;
  setTab: (t: TabKey) => void;
  classes: string[];
  setClasses: (c: string[]) => void;
  stats: SessionStats | null;
  setStats: (s: SessionStats | null) => void;
  selectedImageId: string | null;
  setSelectedImageId: (id: string | null) => void;
}

export const useApp = create<AppState>((set) => ({
  sessionId: null,
  setSessionId: (id) => set({ sessionId: id }),
  tab: "auto",
  setTab: (t) => set({ tab: t }),
  classes: [],
  setClasses: (c) => set({ classes: c }),
  stats: null,
  setStats: (s) => set({ stats: s }),
  selectedImageId: null,
  setSelectedImageId: (id) => set({ selectedImageId: id }),
}));
