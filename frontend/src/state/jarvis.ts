/**
 * Zustand store (PRD §3): everything the DOM HUD needs.
 *
 * The 3D canvas deliberately subscribes to NOTHING here - audio reactivity
 * flows through the mutable `audioLevels` object instead, so log traffic
 * never re-renders the WebGL tree.
 */

import { create } from "zustand";
import type {
  AssistantStatus,
  EngineInfo,
  LogLevel,
  LogLine,
  SettingsState,
  TranscriptTurn,
  Vitals,
} from "@/lib/protocol";

const MAX_LOGS = 220;
const MAX_TURNS = 60;

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
  },
  vitals: null,
  settingsOpen: false,
  transcriptOpen: false,
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

  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  setTranscriptOpen: (transcriptOpen) => set({ transcriptOpen }),

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
