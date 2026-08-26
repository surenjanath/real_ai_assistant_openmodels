# Project J.A.R.V.I.S.

A locally hosted, privacy-first, agentic AI assistant with an audio-reactive
holographic interface and a high-speed multi-agent backend.

- **Interface** — dark, minimalist, geometric: a dim gray wireframe
  icosahedron cage around a dense white particle sphere that pulses, scales
  and accelerates with live speech frequencies; muted `V = 4/3 π r³`
  mathematical anchor at the bottom; floating semi-transparent telemetry
  terminal on the right.
- **Backend** — FastAPI nervous system streaming telemetry logs and raw TTS
  audio over WebSockets, orchestrating a CrewAI-style agent crew on local
  Ollama models, with an n8n + FastMCP automation bridge.
- **Voice** — Kokoro-82M (via `pykokoro`, Core ML/Neural Engine capable)
  streamed as int16 PCM chunks to the browser, driving the Web Audio API
  analyser that animates the hologram.

Everything degrades gracefully: with **zero optional dependencies** the full
product loop still runs (simulated crew + dependency-free synth voice), so you
can develop the interface on any machine and enable the real engines on the
target hardware.

---

## Quickstart

Requirements: Python 3.11+, Node 20+.

```bash
make setup          # backend venv + frontend npm install

make backend        # FastAPI on :8000
make frontend       # Next.js on :3000 (proxies /ws/* and /api/* same-origin)
```

Open **http://localhost:3000**. Boot logs stream into the telemetry panel;
type a directive (e.g. `run a full system diagnostic`) and press Enter — the
crew runs, reasoning lines stream in, and the answer is spoken back with the
particle sphere reacting to the audio.

End-to-end check against a running backend:

```bash
make smoke          # verifies /api/health, /ws/logs, command routing, /ws/audio
```

## What runs where

| Subsystem | Location | Notes |
| --- | --- | --- |
| Holographic scene | `frontend/src/components/Scene.tsx` | R3F canvas: `IcosahedronGeometry` + `MeshBasicMaterial(wireframe)` cage, fibonacci-`BufferGeometry` + `PointsMaterial` particle sphere, additive glow sprite |
| State | `frontend/src/state/jarvis.ts` | Zustand for DOM HUD only; audio levels flow through a mutable singleton so log traffic never re-renders the canvas |
| Web Audio | `frontend/src/audio/engine.ts` | Gapless PCM scheduling, `AnalyserNode` band split (bass/mid/treble) piped into `useFrame` |
| Sockets | `frontend/src/hooks/useJarvisConnection.ts` | `/ws/logs` (telemetry in / commands out), `/ws/audio` (TTS chunks in), auto-reconnect |
| Same-origin proxy | `frontend/server.mjs` | Custom Next.js server proxying `/ws/*` + `/api/*` to the backend — no cross-origin browser calls |
| API layer | `backend/app/main.py` | FastAPI, REST + the two bidirectional WebSockets |
| Telemetry bus | `backend/app/logbus.py`, `telemetry.py` | Async fan-out with backlog; heartbeat simulator (Phase 2) |
| Vocal engine | `backend/app/tts/` | `kokoro_engine.py` (pykokoro → kokoro adapter), `fallback_engine.py` (numpy synth), `manager.py` (threaded streaming, b64 PCM frames) |
| Agent runtime | `backend/app/agents/`, `orchestrator.py` | Ollama probe → live crew (direct or CrewAI) or deterministic simulation; answer routed to TTS |
| Automation | `backend/tools/mcp_server.py`, `deploy/n8n/` | FastMCP allow-listed tool server + n8n compose & starter workflow |
| Desktop wrapper | `frontend/src-tauri/` | Tauri 2 scaffold for a native macOS shell |

## Enabling the real stack (macOS / Apple Silicon)

The backend auto-detects each layer at startup and logs what it chose.

### 1. Local LLMs — Ollama

```bash
brew install ollama && ollama serve
ollama pull llama3.1:8b          # default; override with JARVIS_OLLAMA_MODEL
```

With Ollama reachable at `127.0.0.1:11434`, commands run a real four-agent
crew (Router → OSINT Analyst → Systems Engineer → Synthesizer) with streamed
reasoning logs. Any `JARVIS_OLLAMA_MODEL` works (`qwen3:8b`, `mistral-nemo`,
30B-class models on 64 GB machines, …).

### 2. Natural speech — Kokoro-82M

```bash
pip install pykokoro            # accelerated; Core ML/LiteRT backends
# or the reference implementation:
pip install kokoro>=82M soundfile
```

Selection order: `pykokoro` → `kokoro` → built-in fallback synth.
Target the Neural Engine by isolating the TTS workload:

```bash
JARVIS_TTS_ENGINE=auto JARVIS_TTS_DEVICE=coreml ./run.sh
```

`JARVIS_TTS_DEVICE` is passed through as the pipeline device when the
installed library supports it (pykokoro's LiteRT/Core ML path keeps the GPU
free for LLM inference, per PRD §5). Voice is configurable:
`JARVIS_TTS_VOICE=bm_george` (British male, the JARVIS register) —
`af_heart`, `am_michael`, etc. also work.

### 3. Full agentic framework — CrewAI

```bash
pip install crewai
JARVIS_USE_CREWAI=1 ./run.sh    # real Crew with Ollama LLM + step callbacks
```

Without `crewai` installed, the same four personas run as a direct
Ollama-driven crew with zero extra dependencies.

### 4. Automation — FastMCP + n8n

```bash
make mcp        # stdio FastMCP tool server: exec_command / run_script / speak
make n8n        # n8n on http://localhost:5678 with the voice-bridge workflow
```

Agents' tool calls land on the MCP server (allow-listed commands, audited to
`/tmp/jarvis_mcp_audit.log`); n8n workflows call back into the assistant via
`POST /api/speak` and `POST /api/command`.

### 5. Native macOS app

See `frontend/src-tauri/README.md`.

## WebSocket protocol

`/ws/logs` — client sends `{"type":"command","text":"…","origin":"text|voice"}`
or `{"type":"ping","ts":<ms>}`; server sends `hello`, `log`
(`{level: info|voice|success|warn|error, source, msg}` → gray / blue / green /
amber / red), and `status` (`thinking|speaking|idle`).

`/ws/audio` — server sends `tts.start` (engine, voice, sample_rate), `tts.chunk`
(base64 int16 LE PCM, ~200 ms frames), `tts.end`; client may send
`{"type":"speak","text":"…"}`.

## Configuration (env)

| Variable | Default | Purpose |
| --- | --- | --- |
| `JARVIS_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `JARVIS_OLLAMA_MODEL` | `llama3.1:8b` | Crew model |
| `JARVIS_USE_CREWAI` | `false` | Use real CrewAI crew when installed |
| `JARVIS_TTS_ENGINE` | `auto` | `auto` / `kokoro` / `fallback` |
| `JARVIS_TTS_VOICE` | `bm_george` | Kokoro voice id |
| `JARVIS_TTS_DEVICE` | `auto` | `cpu` / `mps` / `coreml` / `cuda` where supported |
| `JARVIS_TTS_SPEED` | `1.0` | Speech rate |
| `JARVIS_LOG_BACKLOG` | `60` | Log lines replayed to new clients |
| `BACKEND_URL` (frontend) | `http://127.0.0.1:8000` | Proxy target of `server.mjs` |

## Hardware optimisation notes (PRD §4)

- The architecture keeps TTS and LLM inference in separate processes/engines;
  on Apple Silicon point pykokoro at Core ML (Neural Engine) so GPU/Metal
  stays dedicated to Ollama inference across the unified memory pool.
- MLX-framework models slot in via Ollama's MLX registry or by swapping the
  agent runtime's chat call (`backend/app/agents/ollama_runtime.py`) — the
  persona prompts and log plumbing are model-agnostic.
- Streaming is chunked end-to-end (model tokens → sentence logs → 200 ms PCM
  frames → gapless Web Audio scheduling) to keep perceived latency sub-second.

## Implementation phases → code

1. **Visual foundation** — `Scene.tsx`, `MathAnchor.tsx`, `globals.css`
2. **Log streaming** — `logbus.py`, `telemetry.py`, `main.py` (`/ws/logs`), `TelemetryPanel.tsx`
3. **The voice** — `tts/kokoro_engine.py`, `tts/manager.py`, `/ws/audio`
4. **Audio reactivity** — `audio/engine.ts`, `audio/levels.ts`, `useFrame` in `Scene.tsx`
5. **The brain** — `agents/`, `orchestrator.py`, `tools/mcp_server.py`, `deploy/n8n/`
