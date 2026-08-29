/**
 * Zustand store (PRD §3): everything the DOM HUD needs.
 *
 * The 3D canvas deliberately subscribes to NOTHING here - audio reactivity
 * flows through the mutable `audioLevels` object instead, so log traffic
 * never re-renders the WebGL tree.
 */

import { create } from "zustand";
import { applySceneTheme } from "./theme";
import type {
  AssistantStatus,
  LogLevel,
  LogLine,
  MemoryStats,
  MetricsFrame,
  ModelPullFrame,
  SettingsState,
  SkillInfo,
  ToolFrame,
  TranscriptTurn,
  Vitals,
} from "@/lib/protocol";

const MAX_LOGS = 220;
const MAX_TURNS = 60;
const MAX_TOOLS = 40;
const MAX_TOASTS = 4;

/** A transient notice: something the operator should see even if the telemetry
 *  panel is scrolled away or covered by an overlay. */
export interface Toast {
  id: number;
  kind: "error" | "warn" | "success" | "info";
  title: string;
  detail: string;
  ts: number;
}

/** Colour schemes, applied by stamping `data-theme` on <html>. */
export const THEMES = ["arc", "crimson", "emerald", "amber", "violet", "mono"] as const;
export type Theme = (typeof THEMES)[number];

/** Which overlay panel, if any, is open. */
export type Overlay =
  | "none"
  | "settings"
  | "skills"
  | "metrics"
  | "neural"
  | "archive"
  | "help";

interface Engines {
  tts: string;
  ttsLabel: string;
  ttsMode: string;
  agents: string;
  mode: string;
  model: string;
}

interface JarvisState {
  status: AssistantStatus;
  statusDetail: string;
  operator: string;
  logs: LogLine[];
  transcript: TranscriptTurn[];
  /** live-streaming answer text, filled by answer.delta while generating */
  caption: string;
  captionStreaming: boolean;
  engines: Engines;
  settings: SettingsState;
  vitals: Vitals | null;
  settingsOpen: boolean;
  transcriptOpen: boolean;
  overlay: Overlay;
  paletteOpen: boolean;
  theme: Theme;
  /** the running cognitive graph size, for the HUD readout */
  neuralNodes: number;
  neuralEdges: number;
  skills: SkillInfo[];
  tools: ToolFrame[];
  metrics: MetricsFrame | null;
  memoryStats: MemoryStats | null;
  /** bumped whenever the durable store changes underneath an open view, so
   *  the archive reloads instead of showing a conversation that is gone */
  memoryVersion: number;
  /** progress of a model download started from the interface */
  pull: ModelPullFrame | null;
  toasts: Toast[];
  logsConnected: boolean;
  audioConnected: boolean;
  audioUnlocked: boolean;
  latencyMs: number | null;
  micListening: boolean;
  wakeWordOn: boolean;

  pushLog: (level: LogLevel, source: string, msg: string) => void;
  pushTurn: (role: "user" | "assistant", text: string) => void;
  setStatus: (status: AssistantStatus, detail?: string) => void;
  setOperator: (operator: string) => void;
  setEngines: (engines: Partial<Engines>) => void;
  setSettings: (settings: Partial<SettingsState>) => void;
  setVitals: (vitals: Vitals) => void;
  setSettingsOpen: (open: boolean) => void;
  setTranscriptOpen: (open: boolean) => void;
  setOverlay: (overlay: Overlay) => void;
  setPaletteOpen: (open: boolean) => void;
  setTheme: (theme: Theme) => void;
  cycleTheme: () => void;
  setNeuralSize: (nodes: number, edges: number) => void;
  setSkills: (skills: SkillInfo[]) => void;
  pushTool: (tool: ToolFrame) => void;
  setMetrics: (metrics: MetricsFrame) => void;
  setMemoryStats: (stats: MemoryStats) => void;
  bumpMemoryVersion: () => void;
  setPull: (pull: ModelPullFrame | null) => void;
  pushToast: (kind: Toast["kind"], title: string, detail?: string) => void;
  dismissToast: (id: number) => void;
  setConnected: (socket: "logs" | "audio", connected: boolean) => void;
  setAudioUnlocked: (unlocked: boolean) => void;
  setLatency: (ms: number | null) => void;
  setMicListening: (listening: boolean) => void;
  setWakeWord: (on: boolean) => void;
  startCaption: () => void;
  appendCaption: (delta: string) => void;
  setCaption: (text: string) => void;
  clearLogs: () => void;
  clearTranscript: () => void;
}

let logId = 0;
let turnId = 0;
let toastId = 0;

export const useJarvis = create<JarvisState>((set) => ({
  status: "boot",
  statusDetail: "",
  operator: "sir",
  logs: [],
  transcript: [],
  caption: "",
  captionStreaming: false,
  engines: { tts: "…", ttsLabel: "", ttsMode: "", agents: "…", mode: "", model: "" },
  settings: {
    model: "",
    model_verified: false,
    models: [],
    installed: [],
    voice: "",
    voices: [],
    voice_labels: {},
    speed: 1,
    think: false,
    think_supported: false,
    think_active: false,
    persona: "jarvis",
    personas: [],
    tools: true,
    tools_supported: false,
    tools_active: false,
    recall: true,
    volume: 0.9,
    stream_speech: true,
    persisted: [],
  },
  vitals: null,
  settingsOpen: false,
  transcriptOpen: false,
  overlay: "none",
  paletteOpen: false,
  theme: "arc",
  neuralNodes: 0,
  neuralEdges: 0,
  skills: [],
  tools: [],
  metrics: null,
  memoryStats: null,
  memoryVersion: 0,
  pull: null,
  toasts: [],
  logsConnected: false,
  audioConnected: false,
  audioUnlocked: false,
  latencyMs: null,
  micListening: false,
  wakeWordOn: false,

  pushLog: (level, source, msg) =>
    set((state) => {
      const line: LogLine = { id: ++logId, ts: Date.now(), level, source, msg };
      const logs = state.logs.length >= MAX_LOGS ? state.logs.slice(-MAX_LOGS + 1) : state.logs;
      return { logs: [...logs, line] };
    }),

  pushTurn: (role, text) =>
    set((state) => {
      const turn: TranscriptTurn = { id: ++turnId, role, text, ts: Date.now() };
      const prev = state.transcript.length >= MAX_TURNS ? state.transcript.slice(-MAX_TURNS + 1) : state.transcript;
      return { transcript: [...prev, turn] };
    }),

  setStatus: (status, detail = "") =>
    set((state) => ({
      status,
      statusDetail: detail === "" ? state.statusDetail : detail,
    })),

  setOperator: (operator) => set({ operator }),

  setEngines: (engines) => set((state) => ({ engines: { ...state.engines, ...engines } })),

  setSettings: (settings) => set((state) => ({ settings: { ...state.settings, ...settings } })),

  setVitals: (vitals) => set({ vitals }),

  // `settingsOpen` stays the single source of truth for the settings panel so
  // the backend's `ui: open_settings` frame keeps working; the overlay field
  // simply guarantees only one panel is up at a time.
  setSettingsOpen: (settingsOpen) =>
    set((state) => ({
      settingsOpen,
      overlay: settingsOpen ? "settings" : state.overlay === "settings" ? "none" : state.overlay,
    })),
  setTranscriptOpen: (transcriptOpen) => set({ transcriptOpen }),

  setOverlay: (overlay) => set({ overlay, settingsOpen: overlay === "settings" }),
  setPaletteOpen: (paletteOpen) => set({ paletteOpen }),

  setTheme: (theme) => {
    applySceneTheme(theme);
    if (typeof document !== "undefined") {
      document.documentElement.dataset.theme = theme;
      try {
        window.localStorage.setItem("jarvis.theme", theme);
      } catch {
        // Private browsing or blocked storage: the theme simply will not persist.
      }
    }
    set({ theme });
  },

  cycleTheme: () =>
    set((state) => {
      const next = THEMES[(THEMES.indexOf(state.theme) + 1) % THEMES.length];
      applySceneTheme(next);
      if (typeof document !== "undefined") {
        document.documentElement.dataset.theme = next;
        try {
          window.localStorage.setItem("jarvis.theme", next);
        } catch {
          // as above
        }
      }
      return { theme: next };
    }),

  setNeuralSize: (neuralNodes, neuralEdges) => set({ neuralNodes, neuralEdges }),
  setSkills: (skills) => set({ skills }),

  pushTool: (tool) =>
    set((state) => {
      const prev = state.tools.length >= MAX_TOOLS ? state.tools.slice(-MAX_TOOLS + 1) : state.tools;
      return { tools: [...prev, tool] };
    }),

  setMetrics: (metrics) => set({ metrics }),
  setMemoryStats: (memoryStats) => set({ memoryStats }),
  bumpMemoryVersion: () => set((state) => ({ memoryVersion: state.memoryVersion + 1 })),
  setPull: (pull) => set({ pull }),

  pushToast: (kind, title, detail = "") =>
    set((state) => {
      // Never stack the same notice twice - a failing subsystem logs in a
      // tight loop and would otherwise bury the screen in identical cards.
      if (state.toasts.some((t) => t.title === title && t.detail === detail)) return state;
      const toast: Toast = { id: ++toastId, kind, title, detail, ts: Date.now() };
      return { toasts: [...state.toasts, toast].slice(-MAX_TOASTS) };
    }),

  dismissToast: (id) => set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),

  setConnected: (socket, connected) =>
    set(socket === "logs" ? { logsConnected: connected } : { audioConnected: connected }),

  setAudioUnlocked: (audioUnlocked) => set({ audioUnlocked }),

  setLatency: (latencyMs) => set({ latencyMs }),

  setMicListening: (micListening) => set({ micListening }),
  setWakeWord: (wakeWordOn) => set({ wakeWordOn }),

  startCaption: () => set({ caption: "", captionStreaming: true }),
  appendCaption: (delta) => set((state) => ({ caption: (state.caption + delta).slice(-1200) })),
  setCaption: (caption) => set({ caption, captionStreaming: false }),

  clearLogs: () => set({ logs: [] }),
  clearTranscript: () => set({ transcript: [], caption: "" }),
}));
