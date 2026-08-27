# Project J.A.R.V.I.S.

A locally hosted, privacy-first, agentic AI assistant with an audio-reactive
holographic interface and a high-speed multi-agent backend.

- **Interface** — dark, minimalist, geometric: a wireframe icosahedron cage
  around a 24,000-point particle sphere displaced on the GPU by live speech
  frequencies, ringed by an FFT spectrum, an oscilloscope of the waveform,
  tick-marked instrument rings and orbiting satellites; live host vitals on the
  left, telemetry terminal on the right, spoken captions along the bottom.
- **Backend** — FastAPI nervous system streaming telemetry, real host metrics
  and raw TTS audio over WebSockets, orchestrating local Ollama models, with an
  n8n + FastMCP automation bridge.
- **Voice** — Kokoro-82M via `pykokoro` (ONNX, Core ML capable), streamed
  sentence-by-sentence as int16 PCM to the browser, driving the Web Audio
  analyser that animates the hologram.

Everything degrades gracefully: with **zero optional dependencies** the full
product loop still runs (simulated crew + dependency-free synth voice), so you
can develop the interface on any machine and enable the real engines on the
target hardware.

---

## Quickstart

Requirements: **Python 3.12**, Node 20+.

> ⚠️ **Python 3.12 specifically.** The Kokoro voice stack has no wheels for
> 3.13+ (`kokoro` requires `<3.13`, `kokoro-onnx` requires `<3.14`). On a newer
> interpreter the backend still runs but silently falls back to the robotic
> synth voice. `make setup` pins 3.12 for you (via [uv](https://astral.sh/uv)
> if installed) and fails loudly otherwise.

```bash
make setup          # backend venv (python 3.12) + frontend npm install

make backend        # FastAPI on :8000
make frontend       # Next.js on :3000 (proxies /ws/* and /api/* same-origin)
```

Open **http://localhost:3000**. Boot logs stream into the telemetry panel;
type a directive (or use the mic) and the answer is spoken back with the
particle sphere reacting to the audio in real time.

The first launch downloads the Kokoro weights (~330 MB) into the Hugging Face
cache and warms the ONNX session in the background — the log says
`vocal engine warm in …` when speech is ready.

### Speed: extended thinking is off by default

Modern local models (qwen3.x, deepseek-r1, gemma, …) emit a long
chain-of-thought before answering. Measured on this project with `qwen3.5:9b`:

| Extended thinking | Time to a spoken answer |
| --- | --- |
| **off** (default) | **~2.5 s** |
| on | ~95 s |

So it ships **off**, and you opt in when you actually want deliberation —
toggle it in the settings panel, or say *"think harder"* / *"answer faster"*.
When it is on, the reasoning is routed to the telemetry panel as
`agent/thought` lines and never spoken aloud. Set `JARVIS_THINK=1` to default
it on.

### Voice control

Say (mic button) or type any of these — handled directly, before the crew:

| Voice command | Effect |
| --- | --- |
| `settings` | Opens the settings panel and speaks the current configuration |
| `status` | Speaks engine, model, and thinking state |
| `list models` / `list voices` | Enumerates what is actually installed |
| `switch model to qwen3 8b` | Live-switches the crew model (spoken tag form → `qwen3:8b`) |
| `change voice to af heart` | Hot-swaps the Kokoro voice (and its language) |
| `speak faster` / `set speed to 1.2` | Adjusts speech rate (0.5–2.0×) |
| `think harder` / `answer faster` | Enables / disables extended thinking |
| `stop` | Barge-in — cuts off speech immediately |
| `new conversation` | Clears short-term memory |
| `what time is it` / `what's the date` | Answered locally, no model call |
| `clear the log`, `repeat that`, `who are you`, `help` | As they read |

Control phrases are **anchored**: an ordinary question like *"how do I make
this loop run faster?"* goes to the model rather than being swallowed as a
speech-rate command.

### Keyboard

| Key | Action |
| --- | --- |
| `/` | Focus the command bar |
| `Esc` | Barge-in — stop speaking |
| `⌘K` / `Ctrl+K` | Toggle wake-word listening |
| `↑` / `↓` | Command history |

### Wake word

Click **WAKE** (or `⌘K`) for continuous listening: J.A.R.V.I.S. ignores the
room until it hears *"Jarvis, …"*. An echo guard drops anything the microphone
picks up while the assistant is speaking, so it never answers its own voice —
unless you deliberately say the wake word to interrupt.

### End-to-end check

```bash
make smoke          # health, vitals, settings, intents, barge-in, real audio
```

`smoke` asserts the audio stream is actually **audible** (non-trivial duration
and peak amplitude) rather than merely present, so a silent regression fails
instead of passing quietly. Add `--wav out.wav` to keep the sample and judge
the voice by ear.

## What runs where

| Subsystem | Location | Notes |
| --- | --- | --- |
| Holographic scene | `frontend/src/components/Scene.tsx` | R3F canvas: GPU-displaced 24k-point core (`ShaderMaterial`), 64-band FFT bars, 128-point waveform ring, tick-marked HUD rings, gyroscope rings, radar sweep, satellites with trails |
| State | `frontend/src/state/jarvis.ts` | Zustand for DOM HUD only; audio levels flow through a mutable singleton so log traffic never re-renders the canvas |
| Web Audio | `frontend/src/audio/engine.ts` | Gapless PCM scheduling, analyser (bands + waveform + transient kick), barge-in flush, autoplay-state tracking |
| Sockets | `frontend/src/hooks/useJarvisConnection.ts` | `/ws/logs` (telemetry + commands), `/ws/audio` (TTS chunks), auto-reconnect |
| Speech input | `frontend/src/hooks/useSpeechRecognition.ts` | Push-to-talk + continuous wake-word mode with auto-restart |
| Vitals | `backend/app/vitals.py`, `frontend/src/components/VitalsPanel.tsx` | Real psutil metrics → arc gauges + sparklines |
| Settings | `backend/app/registry.py`, `frontend/src/components/SettingsPanel.tsx` | Live model/voice/speed/thinking registry; `GET/POST /api/settings` |
| Same-origin proxy | `frontend/server.mjs` | Custom Next.js server proxying `/ws/*` + `/api/*` — no cross-origin calls |
| API layer | `backend/app/main.py` | FastAPI, REST + two bidirectional WebSockets |
| Telemetry bus | `backend/app/logbus.py`, `telemetry.py` | Thread-safe async fan-out; heartbeats derived from real metrics |
| Vocal engine | `backend/app/tts/` | `kokoro_engine.py` (pykokoro → kokoro), `fallback_engine.py`, `manager.py` (threaded streaming, playback gating, barge-in) |
| Agent runtime | `backend/app/agents/`, `orchestrator.py` | Ollama probe → fast/crew/crewai runtimes, `<think>` filtering, conversational memory |
| Automation | `backend/tools/mcp_server.py`, `deploy/n8n/` | FastMCP allow-listed tool server + n8n compose & starter workflow |
| Desktop wrapper | `frontend/src-tauri/` | Tauri 2 scaffold for a native macOS shell |

## Enabling the real stack (macOS / Apple Silicon)

The backend auto-detects each layer at startup and logs what it chose.

### 1. Local LLMs — Ollama

```bash
brew install ollama && ollama serve
ollama pull qwen3:8b            # any tag works
```

With Ollama reachable at `127.0.0.1:11434` the crew runs live. If
`JARVIS_OLLAMA_MODEL` is unset (or names a model you have not pulled), the
backend **auto-selects an installed model**, preferring small-and-fast ones,
and says so in the log. The settings panel separates *Installed* from
*Not installed* so you cannot silently pick a model that cannot answer.

Modes (`JARVIS_CREW_MODE`):

- `fast` (default) — one streamed call in the J.A.R.V.I.S. persona with
  short-term memory. Conversational latency.
- `crew` — the four-persona pipeline (Router → Analyst → Engineer →
  Synthesiser), each step streamed to the telemetry panel.
- `JARVIS_USE_CREWAI=1` — the same crew built with the real `crewai` package.

### 2. Natural speech — Kokoro-82M

```bash
pip install 'pykokoro[coreml]'   # ONNX, no torch; Core ML capable
```

Selection order: `pykokoro` → `kokoro` → built-in fallback synth.

```bash
JARVIS_TTS_DEVICE=coreml JARVIS_TTS_QUALITY=fp32 ./run.sh
```

Measured on an M1 Max (24 kHz, per sentence):

| Provider | Quality | Real-time factor | First sentence |
| --- | --- | --- | --- |
| cpu | fp32 | 0.43 | 1.3 s |
| coreml | fp32 | 0.48 | 1.7 s |
| cpu | q8 | 1.04 | 3.6 s |

`fp32` is **~2.4× faster than `q8`** here despite the larger download, so it is
the default. Core ML keeps the work off the GPU so Metal stays free for Ollama
inference; plain `cpu` is marginally faster in wall-clock terms — pick per your
workload. Speech is synthesised **sentence by sentence**, so the first sentence
starts playing while the rest is still being generated.

Voice is configurable: `JARVIS_TTS_VOICE=bm_george` (British male, the JARVIS
register). Switching voice also switches the G2P language automatically —
`af_*` → en-US, `bm_*` → en-GB, and so on.

### 3. Automation — FastMCP + n8n

```bash
make mcp        # stdio FastMCP tool server: exec_command / run_script / speak
make n8n        # n8n on http://localhost:5678 with the voice-bridge workflow
```

Agents' tool calls land on the MCP server (allow-listed commands, audited to
`/tmp/jarvis_mcp_audit.log`); n8n workflows call back via `POST /api/speak` and
`POST /api/command`.

### 4. Native macOS app

See `frontend/src-tauri/README.md`.

## WebSocket protocol

`/ws/logs` — client sends `{"type":"command","text":…,"origin":"text|voice"}`,
`{"type":"stop"}` or `{"type":"ping","ts":…}`; server sends:

| Frame | Meaning |
| --- | --- |
| `hello` | Name, version, engines, settings snapshot, first vitals |
| `log` | `{level: info\|voice\|success\|warn\|error, source, msg}` → gray / blue / green / amber / red |
| `status` | `thinking \| speaking \| idle` |
| `vitals` | Live host metrics (cpu, mem, disk, net, load, battery) |
| `answer.start` / `answer.delta` / `answer` | The reply, streamed token-by-token for the caption |
| `transcript` | A completed conversational turn |
| `settings.update` | After any model / voice / speed / thinking change |
| `ui` | `open_settings`, `close_settings`, `clear_logs`, `clear_transcript` |

`/ws/audio` — server sends `tts.start` (engine, voice, sample_rate, text),
`tts.chunk` (base64 int16 LE PCM, ~200 ms frames), `tts.end`, `tts.flush`
(barge-in) and `tts.error`; client may send `{"type":"speak","text":…}` or
`{"type":"stop"}`. It is a **broadcast** bus — always match on `utterance_id`.

## Configuration (env)

| Variable | Default | Purpose |
| --- | --- | --- |
| `JARVIS_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `JARVIS_OLLAMA_MODEL` | *(auto-select)* | Crew model |
| `JARVIS_CREW_MODE` | `fast` | `fast` / `crew` |
| `JARVIS_THINK` | `false` | Extended chain-of-thought |
| `JARVIS_MEMORY_TURNS` | `8` | Conversational turns retained |
| `JARVIS_USE_CREWAI` | `false` | Use real CrewAI crew when installed |
| `JARVIS_TTS_ENGINE` | `auto` | `auto` / `kokoro` / `fallback` |
| `JARVIS_TTS_VOICE` | `bm_george` | Kokoro voice id |
| `JARVIS_TTS_DEVICE` | `auto` | ONNX provider: `cpu` / `coreml` / `cuda` / … |
| `JARVIS_TTS_QUALITY` | `fp32` | `fp32` / `q8` / `q4` … |
| `JARVIS_TTS_SPEED` | `1.0` | Speech rate |
| `JARVIS_TTS_WARMUP` | `true` | Pay ONNX session init at boot |
| `JARVIS_OPERATOR` | `sir` | How the assistant addresses you |
| `JARVIS_VITALS_INTERVAL_S` | `2.0` | Host metrics sample period |
| `JARVIS_LOG_BACKLOG` | `60` | Log lines replayed to new clients |
| `BACKEND_URL` (frontend) | `http://127.0.0.1:8000` | Proxy target of `server.mjs` |

## Hardware optimisation notes (PRD §4)

- TTS and LLM inference stay in separate engines; pointing pykokoro at Core ML
  keeps Metal dedicated to Ollama across the unified memory pool.
- Particle displacement runs in a **vertex shader**, so the 24,000-point core
  costs the CPU nothing per frame and the main thread stays free for
  WebSocket traffic.
- Streaming is chunked end-to-end (model tokens → sentence logs → per-sentence
  synthesis → 200 ms PCM frames → gapless Web Audio scheduling) to keep
  perceived latency low.

## Implementation phases → code

1. **Visual foundation** — `Scene.tsx`, `MathAnchor.tsx`, `globals.css`
2. **Log streaming** — `logbus.py`, `telemetry.py`, `vitals.py`, `main.py`, `TelemetryPanel.tsx`
3. **The voice** — `tts/kokoro_engine.py`, `tts/manager.py`, `/ws/audio`
4. **Audio reactivity** — `audio/engine.ts`, `audio/levels.ts`, shaders in `Scene.tsx`
5. **The brain** — `agents/`, `orchestrator.py`, `intents.py`, `tools/mcp_server.py`, `deploy/n8n/`
