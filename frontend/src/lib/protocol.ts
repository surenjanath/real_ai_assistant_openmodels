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

export interface PersonaInfo {
  key: string;
  label: string;
  blurb: string;
  temperature: number;
  /** the voice this disposition speaks in — selecting it adopts this */
  voice?: string;
  speed?: number;
  /** the prompt that defines the manner; editable by the operator */
  style?: string;
  /** ships with the assistant, so it can be reset but never deleted */
  builtin?: boolean;
  /** a built-in the operator has changed — offer reset rather than delete */
  edited?: boolean;
  /** entirely the operator's own — deletable */
  custom?: boolean;
}

export interface SkillInfo {
  name: string;
  description: string;
  /** safe | reads_files | executes | network */
  danger: string;
  params: string[];
}

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
  /** selected personality preset */
  persona: string;
  personas: PersonaInfo[];
  /** whether the cortex may call skills natively */
  tools: boolean;
  /** whether the current model advertises tool calling at all */
  tools_supported: boolean;
  tools_active: boolean;
  /** durable cross-session recall */
  recall: boolean;
  /** playback gain 0..1 */
  volume: number;
  /** speak each sentence as it is written, rather than waiting for the answer */
  stream_speech: boolean;
  /** which settings are saved to disk and will be restored on the next boot */
  persisted?: string[];
  /** where that preference file lives */
  prefs_path?: string;
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

export interface MemoryStats {
  turns: number;
  sessions: number;
  facts: number;
  notes: number;
  reminders: number;
  since: number | null;
  path: string;
  fts: boolean;
  size_kb: number;
}

export interface HelloFrame {
  type: "hello";
  name: string;
  version: string;
  operator?: string;
  engines: EngineInfo;
  settings?: SettingsState;
  vitals?: Vitals;
  skills?: SkillInfo[];
  memory?: MemoryStats | null;
}

/* ---- the cognitive graph (backend/app/neural.py) ---- */

export interface NeuralNodeFrame {
  id: string;
  label: string;
  /** sensory | intake | memory | cortex | effector | motor */
  region: string;
  layer: number;
  /** core | tool */
  kind: string;
}

export interface NeuralEdgeFrame {
  id: string;
  from: string;
  to: string;
  weight: number;
}

/** Sent once per client, and again whenever a tool node is grown. */
export interface NeuralGraphFrame {
  type: "neural.graph";
  nodes: NeuralNodeFrame[];
  edges: NeuralEdgeFrame[];
  levels?: number[];
  fired?: number;
}

/** Coalesced activation, ~20 Hz while anything is firing. */
export interface NeuralFrame {
  type: "neural";
  ts: number;
  /** [nodeIndex, intensity] pairs that spiked since the last frame */
  spikes: Array<[number, number]>;
  /** [edgeIndex, intensity] pairs that carried a signal */
  flows: Array<[number, number]>;
  /** decayed activation per node, index-aligned with the graph */
  levels: number[];
  regions: Record<string, number>;
  fired: number;
}

/* ---- instrumentation ---- */

export interface MetricsBand {
  p50: number;
  p95: number;
  last: number;
}

export interface MetricsRun {
  text: string;
  origin: string;
  model: string;
  mode: string;
  kind: string;
  ttft_ms: number;
  /** time to the first *spoken* word - what the operator actually waits for */
  ttfa_ms: number | null;
  total_ms: number;
  voice_ms: number | null;
  tok_s: number;
  chars: number;
  /** prompt tokens Ollama evaluated — what a slow first word usually is */
  prompt_tokens?: number;
  eval_tokens?: number;
  tool_calls: number;
  tools_used: string[];
  error: string;
}

export interface MetricsFrame {
  type: "metrics";
  ts: number;
  commands: number;
  errors: number;
  tool_calls: number;
  spoken_chars: number;
  uptime_s: number;
  ttft_ms: MetricsBand;
  ttfa_ms?: MetricsBand;
  total_ms: MetricsBand;
  tok_s: { avg: number; best: number; last: number };
  last: MetricsRun | null;
  history: Array<{
    ttft_ms: number;
    ttfa_ms?: number | null;
    total_ms: number;
    tok_s: number;
    kind: string;
    error: boolean;
  }>;
}

/** One executed skill, as it happens. */
export interface ToolFrame {
  type: "tool";
  name: string;
  args: Record<string, unknown>;
  ok: boolean;
  detail: string;
  elapsed_ms: number;
  ts: number;
}

/** Progress of an `ollama pull` started from the interface. */
export interface ModelPullFrame {
  type: "model.pull";
  model: string;
  status?: string;
  completed?: number;
  total?: number;
  percent?: number | null;
  done: boolean;
  ok?: boolean;
  detail?: string;
}

export interface ReminderFrame {
  type: "reminder" | "reminder.set";
  id: number;
  text: string;
  due_ts: number;
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

/** Something in the durable store changed underneath any open archive view. */
export interface MemoryChangedFrame {
  type: "memory.changed";
  reason: string;
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
  | NeuralGraphFrame
  | NeuralFrame
  | MetricsFrame
  | ModelPullFrame
  | ToolFrame
  | ReminderFrame
  | MemoryChangedFrame
  | { type: "pong"; ts?: number };

export interface CommandFrame {
  type: "command" | "speak" | "stop";
  text?: string;
  origin?: "text" | "voice";
  /** The client has acoustic evidence that a person, not the assistant's own
   *  playback, produced this transcript — so the backend's content-only echo
   *  guard should stand down. See `frontend/src/audio/mic.ts`. */
  verified?: boolean;
}

/** Settings delta pushed over the telemetry socket (same shape as POST /api/settings). */
export interface SettingsCommandFrame {
  type: "settings";
  model?: string;
  voice?: string;
  speed?: number;
  think?: boolean;
  persona?: string;
  tools?: boolean;
  recall?: boolean;
  volume?: number;
  stream_speech?: boolean;
}

/** Invoke a skill by hand from the interface. */
export interface SkillCommandFrame {
  type: "skill";
  name: string;
  arguments?: Record<string, unknown>;
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
