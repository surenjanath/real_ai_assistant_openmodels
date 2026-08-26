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

/* ---- /ws/logs ---- */

export interface HelloFrame {
  type: "hello";
  name: string;
  version: string;
  engines: { tts: string; tts_mode: string; agents: string; model: string };
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

export type LogsServerFrame = HelloFrame | LogFrame | StatusFrame | { type: "pong"; ts: number };

export interface CommandFrame {
  type: "command" | "speak";
  text: string;
  origin: "text" | "voice";
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

export type AudioServerFrame = TtsStartFrame | TtsChunkFrame | TtsEndFrame | TtsErrorFrame;
