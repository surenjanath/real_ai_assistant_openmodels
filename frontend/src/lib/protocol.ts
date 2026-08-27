/** Wire protocol shared with the FastAPI backend (backend/app/main.py). */

export type LogLevel = "info" | "voice" | "success" | "warn" | "error";

export type AssistantStatus = "boot" | "idle" | "listening" | "thinking" | "speaking";

export interface LogLine {
  id: number;
  ts: number;
  level: LogLevel;
  source: string;
  msg: string;
}

export interface TranscriptTurn {
  id: number;
  role: "user" | "assistant";
  text: string;
  ts: number;
}

/* ---- /ws/logs ---- */

export interface SettingsState {
  model: string;
  model_verified: boolean;
  models: string[];
  /** tags Ollama actually reports - the only source of truth for "installed" */
  installed: string[];
  voice: string;
  voices: string[];
  voice_labels: Record<string, string>;
  speed: number;
  /** extended chain-of-thought, opt-in (slow but more careful) */
  think: boolean;
  /** whether the current model has a thinking mode at all */
  think_supported: boolean;
  think_active: boolean;
}

export interface Vitals {
  cpu: number;
  mem: number;
  mem_used_gb?: number;
  mem_total_gb?: number;
  disk: number;
  disk_free_gb?: number;
  net_kbps: number;
  load: number[];
  cores: number;
  procs?: number;
  battery?: number;
  power?: "ac" | "battery";
  uptime_s?: number;
  clients?: number;
  source?: string;
}

export interface EngineInfo {
  tts: string;
  tts_label?: string;
  tts_mode: string;
  agents: string;
  mode?: string;
  model: string;
}

export interface HelloFrame {
  type: "hello";
  name: string;
  version: string;
  operator?: string;
  engines: EngineInfo;
  settings?: SettingsState;
  vitals?: Vitals;
}

export interface LogFrame {
  type: "log";
  ts: number;
  level: LogLevel;
  source: string;
  msg: string;
}

export interface StatusFrame {
  type: "status";
  status: AssistantStatus | string;
  detail?: string;
}

export interface SettingsUpdateFrame {
  type: "settings.update";
  settings: SettingsState;
}

export interface VitalsFrame extends Vitals {
  type: "vitals";
  ts: number;
}

/** Streamed answer tokens, so the caption fills in as the model generates. */
export interface AnswerStartFrame {
  type: "answer.start";
}
export interface AnswerDeltaFrame {
  type: "answer.delta";
  text: string;
}
export interface AnswerFrame {
  type: "answer";
  text: string;
}

export interface TranscriptFrame {
  type: "transcript";
  role: "user" | "assistant";
  text: string;
}

export interface UiFrame {
  type: "ui";
  action: "open_settings" | "close_settings" | "clear_logs" | "clear_transcript" | string;
}

export type LogsServerFrame =
  | HelloFrame
  | LogFrame
  | StatusFrame
  | SettingsUpdateFrame
  | VitalsFrame
  | AnswerStartFrame
  | AnswerDeltaFrame
  | AnswerFrame
  | TranscriptFrame
  | UiFrame
  | { type: "pong"; ts?: number };

export interface CommandFrame {
  type: "command" | "speak" | "stop";
  text?: string;
  origin?: "text" | "voice";
}

/* ---- /ws/audio ---- */

export interface TtsStartFrame {
  type: "tts.start";
  utterance_id: string;
  engine: string;
  voice: string;
  sample_rate: number;
  text: string;
}

export interface TtsChunkFrame {
  type: "tts.chunk";
  utterance_id: string;
  seq: number;
  sample_rate: number;
  /** base64-encoded int16 little-endian mono PCM */
  data: string;
}

export interface TtsEndFrame {
  type: "tts.end";
  utterance_id: string;
  frames?: number;
  duration_s?: number;
  cancelled?: boolean;
}

export interface TtsErrorFrame {
  type: "tts.error";
  utterance_id: string;
  detail: string;
}

/** Barge-in: drop anything buffered but not yet played. */
export interface TtsFlushFrame {
  type: "tts.flush";
  reason: string;
}

export type AudioServerFrame =
  | TtsStartFrame
  | TtsChunkFrame
  | TtsEndFrame
  | TtsErrorFrame
  | TtsFlushFrame;
