/**
 * Zustand store (PRD §3): everything the DOM HUD needs.
 *
 * The 3D canvas deliberately subscribes to NOTHING here - audio reactivity
 * flows through the mutable `audioLevels` object instead, so log traffic
 * never re-renders the WebGL tree.
 */

import { create } from "zustand";
import type { AssistantStatus, LogLevel, LogLine } from "@/lib/protocol";

const MAX_LOGS = 160;

interface Engines {
  tts: string;
  ttsMode: string;
  agents: string;
  model: string;
}

interface JarvisState {
  status: AssistantStatus;
  statusDetail: string;
  logs: LogLine[];
  engines: Engines;
  logsConnected: boolean;
  audioConnected: boolean;
  latencyMs: number | null;

  pushLog: (level: LogLevel, source: string, msg: string) => void;
  setStatus: (status: AssistantStatus, detail?: string) => void;
  setEngines: (engines: Partial<Engines>) => void;
  setConnected: (socket: "logs" | "audio", connected: boolean) => void;
  setLatency: (ms: number | null) => void;
  clearLogs: () => void;
}

let logId = 0;

export const useJarvis = create<JarvisState>((set) => ({
  status: "boot",
  statusDetail: "",
  logs: [],
  engines: { tts: "…", ttsMode: "", agents: "…", model: "" },
  logsConnected: false,
  audioConnected: false,
  latencyMs: null,

  pushLog: (level, source, msg) =>
    set((state) => {
      const line: LogLine = { id: ++logId, ts: Date.now(), level, source, msg };
      const logs = state.logs.length >= MAX_LOGS ? state.logs.slice(-MAX_LOGS + 1) : state.logs;
      return { logs: [...logs, line] };
    }),

  setStatus: (status, detail = "") =>
    set((state) => ({
      status,
      statusDetail: detail === "" ? state.statusDetail : detail,
    })),

  setEngines: (engines) =>
    set((state) => ({ engines: { ...state.engines, ...engines } })),

  setConnected: (socket, connected) =>
    set(socket === "logs" ? { logsConnected: connected } : { audioConnected: connected }),

  setLatency: (latencyMs) => set({ latencyMs }),

  clearLogs: () => set({ logs: [] }),
}));
