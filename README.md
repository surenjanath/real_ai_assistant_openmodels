# Project J.A.R.V.I.S.

A locally hosted, privacy-first, agentic AI assistant with an audio-reactive
holographic interface and a high-speed multi-agent backend.

- **Interface** — dark, minimalist, geometric: a wireframe icosahedron cage
  around a 24,000-point particle sphere displaced on the GPU by live speech
  frequencies, ringed by an FFT spectrum, an oscilloscope of the waveform,
  tick-marked instrument rings and orbiting satellites; live host vitals on the
  left, telemetry terminal on the right, spoken captions along the bottom.
- **Neural system** — the assistant's own cognitive architecture, drawn as a
  living network: every subsystem is a node, every hand-off an edge, and real
  action potentials travel the synapses as a directive passes through. Nothing
  pulses unless that code path actually ran.
- **Backend** — FastAPI nervous system streaming telemetry, real host metrics,
  neural activation and raw TTS audio over WebSockets, orchestrating local
  Ollama models with native tool calling, durable on-device memory and a
  deterministic reflex arc; plus an n8n + FastMCP automation bridge.
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


### Speed: the assistant speaks before it has finished thinking

The vocal engine is fastest when handed whole sentences — Kokoro synthesises a
sentence in about a third of the time it takes to say it. So the answer is not
waited for: `backend/app/tts/segmenter.py` watches the token stream, and the
moment a sentence closes it goes straight to the vocal engine while the model
is still writing the next one. The first fragment is even allowed to break at a
clause, because Kokoro emits no audio until a whole sentence is done and a
35-word opening sentence is several seconds of silence.

Measured on this project with `qwen3.5:9b`, three-sentence answers, as the gap
between the answer being composed and the first word being *heard*:

| Speech start | Delay after the answer is composed |
| --- | --- |
| **streamed** (default) | **~0.6 s** — and sometimes negative: it is already talking |
| whole-answer | ~1.9 s |

The gain grows with answer length, since the first sentence closes at the same
early moment however long the reply turns out to be. Toggle it in the settings
panel under **SPEECH START**, or set `JARVIS_STREAM_SPEECH=0`.

Two related costs are paid up front rather than on your first directive:

* **Model residency.** Paging an 8B model in costs 2–10 s. The backend preloads
  it at boot and again the instant you switch model (`JARVIS_PRELOAD`), and
  sends `keep_alive` so Ollama holds it for the session
  (`JARVIS_OLLAMA_KEEP_ALIVE`, default `30m`).
* **Vocal warmup.** The ONNX session is built at boot (`JARVIS_TTS_WARMUP`).

The interface reports both figures: **FIRST WORD** in the instrument bar is the
time to the first *spoken* word, not the first generated token — open the
performance view for the p50/p95 of each.


### Render quality adapts to the machine

The hologram is 24,000 GPU-displaced particles at full quality, which not every
machine can sustain — and a scene that drops frames also starves the audio
callback driving it, so the sphere stops reacting to the voice. The tier is
picked from what the device reports, then corrected by the frame rate actually
measured (`frontend/src/state/quality.ts`); it steps down at most once per
session, never oscillating. Override it under **RENDER QUALITY** in the
settings panel — the choice is remembered on that device.

| Tier | Particles | Stars | DPR cap | Antialias |
| --- | --- | --- | --- | --- |
| high | 24,000 | 520 | 2 | yes |
| balanced | 12,000 | 340 | 1.5 | yes |
| low | 5,000 | 160 | 1 | no |


## The neural system

The interface's centrepiece is no longer only audio-reactive — it is
*cognition*-reactive. `backend/app/neural.py` models the running assistant as a
directed graph:

| Region | Nodes | Fires when |
| --- | --- | --- |
| sensory | TEXT IN, SPEECH IN, HOST SENSE | a directive arrives, or vitals are sampled |
| intake | INTENT, CONTROL, ROUTER, REFLEX | the command is classified and grounded |
| memory | WORKING, RECALL, CONSOLIDATE | short-term context, durable recall, persistence |
| cortex | PERSONA, ANALYST, ENGINEER, SYNTH, DELIBERATE | the model actually generating |
| effector | TOOL BUS, plus one node per skill | a skill is invoked |
| motor | COMPOSE, VOCALISE, AUDIO OUT | the answer is composed and spoken |

Every hand-off between stages emits an action potential that travels the
corresponding synapse in the 3D shell, and a node's brightness is its live
activation. **A tool grows its own node the first time it is really called**, so
the graph is a record of exercised capability rather than a catalogue of
intentions.

Activation is coalesced to 20 Hz on the server (a streaming model fires the
cortex thousands of times a second) and lands in a mutable singleton on the
client, so the neural stream never re-renders React or the WebGL tree.

The `NEURAL CORTEX` panel on the left summarises the same activity as six
region meters plus a spike odometer; clicking it opens the **cognitive map** —
every node, named and grouped by region, with its live activation. Both are
painted from their own animation frame straight into the DOM, so twenty
activation frames a second never pass through React.

## Settings survive the restart

Model, voice, speed, persona, skills, recall, volume and speech streaming are
written through to `~/.jarvis/settings.json` the moment you change them, and
restored at boot. Environment variables still win at *first* boot, when there
is nothing saved yet; after that the saved value is authoritative, because it
represents a deliberate act rather than a shell default.

Two things are re-validated rather than trusted: `think` and `tools` are model
*capabilities*, so a preference saved against one model is stood down (with a
log line) if the model loaded now cannot do it. A corrupt or unwritable
settings file costs persistence, never the boot — it degrades to session-only
settings and says so.

`DELETE /api/settings` erases the file; the live session keeps its settings and
the *next* boot falls back to the environment.

## The archive

Every conversation and every skill invocation has been recorded since the
hippocampus was added, but neither had a way in. `⌗` in the telemetry header —
or `⌘P` → *archive* — opens both:

* **Conversations** — each session, newest first, grouped by day and named
  after its opening directive. Search finds a conversation only when you
  already remember a word from it; what people actually remember is *when*.
  Expand one to read it back, or erase it (two clicks, no undo).
* **Audit trail** — every skill that has actually run, with the arguments it
  was called with and how long it took. This thing reads files, writes files
  and, when armed, makes network requests on your machine; a record of that
  which can only be read with `sqlite3` is not a record anybody checks.

## Skills — things it can actually do

When the selected model advertises the `tools` capability, the skill schemas
are attached to every request and the runtime runs the full loop: model →
`tool_calls` → execute locally → feed results back → model.

| Skill | Does |
| --- | --- |
| `get_datetime` | exact local date, time, weekday, timezone |
| `calculate` | precise arithmetic (AST-evaluated, never `eval`) |
| `convert_units` | length, mass, time, volume, data, speed, temperature |
| `text_stats` | words, sentences, reading and speaking time, top terms |
| `pick_random` | fair dice, coins, draws and picks — a model cannot do this itself |
| `system_status` | live CPU, memory, disk, network, battery, uptime |
| `list_processes` | what is actually eating this machine's CPU or RAM |
| `remember` / `recall` | durable facts and search across every past session |
| `take_note` / `list_notes` | free-form notes |
| `set_reminder` / `list_reminders` | spoken reminders on a scheduler |
| `list_directory` / `read_file` / `search_files` | the permitted workspace only |
| `directory_size` | where the disk has gone, largest children first |
| `write_file` / `list_scratch_files` | text files, in its own scratch folder only |
| `run_command` | allow-listed read-only shell — **off** unless `JARVIS_ALLOW_SHELL=1` |
| `web_search` | current results with links — **off** unless `JARVIS_ALLOW_NET=1` |
| `get_weather` | conditions and a 3-day forecast — **off** unless `JARVIS_ALLOW_NET=1` |
| `fetch_url` | readable page text — **off** unless `JARVIS_ALLOW_NET=1` |

Safety posture, in four boundaries:

* **Reads** are confined to `JARVIS_WORKSPACE` (default `~`) with a denylist for
  `.ssh`, `.aws`, `.env`, keychains and friends.
* **Writes** land only in `~/.jarvis/files`, text files only, and the path is
  re-checked *after* resolution so neither `..` nor a planted symlink escapes.
  Nothing the user already had can be overwritten.
* **Shell** is off unless `JARVIS_ALLOW_SHELL=1`, and then only allow-listed
  read-only verbs, argv-form, with a timeout.
* **Network** is off unless `JARVIS_ALLOW_NET=1`, and then every URL is resolved
  before the request and refused if it points at this machine or a private
  network — otherwise "fetch a URL" reaches the cloud metadata endpoint, the
  Ollama admin API on `:11434`, and everything else that trusts localhost.
  Each redirect hop is re-checked, because a `302` would otherwise walk
  straight past the first check.

Every invocation is audited to the memory store and shown in the archive's
**audit trail**. `make test` asserts each boundary refuses; `make smoke`
asserts the sandbox refuses to escape against a running server.

### The reflex arc

A tool-capable model *decides* whether to call a tool, and that decision is
unreliable exactly where it matters. Measured here on `qwen3.5:9b`, with the
strongest prompt wording tried and the `calculate` schema attached:

| Setting | Called the tool for `48271 × 9912` |
| --- | --- |
| temperature 0.6 | 4 / 6 |
| temperature 0.3 | 2 / 6 |
| temperature 0.1 | 3 / 6 |

Every run where it declined produced a *different* wrong number
(478,464,152 / 478,463,152 / 478,462,352 — the answer is **478,462,152**).
Lowering the temperature did not help, because it is a judgement failure rather
than a sampling one.

So arithmetic is not left to judgement. `backend/app/reflexes.py` is a cheap,
deterministic pre-flight: if a directive plainly depends on something this
machine can compute exactly — arithmetic on numbers longer than two digits,
live host state, the current date — it is computed *first* and injected as
grounding. The model still writes the prose; it simply can no longer get the
number wrong. With the reflex in place the same question answered correctly on
3 of 3 runs.

Reflexes are deliberately conservative: `2 + 2` and "tell me about the Roman
empire" ground nothing, because noise in the context window costs latency on
every turn.

## Memory that survives a restart

`backend/app/memory.py` is a local SQLite hippocampus at `~/.jarvis/jarvis.db`
(override with `JARVIS_DATA_DIR`):

- **episodic** — every turn, with model, latency and session
- **semantic** — facts you asked it to remember, keyed so corrections overwrite
- **notes** and **reminders** — the reminder scheduler speaks them when due,
  waiting for a gap rather than talking over an utterance in flight
- **recall** — keyword-ranked over FTS5 where available, a scored scan otherwise

Relevant fragments are injected ahead of each directive. Two details matter:
the turn being answered is excluded from its own recall (otherwise the
assistant "remembers" the question it was asked one second ago), and recalled
text is framed as *a record of what was said, not a source of truth* — an
earlier answer of its own may have been wrong, and repeating it would launder a
hallucination into a fact.

The `MIND` tab browses all of it, and searches every past session. Nothing
leaves the machine; `DELETE /api/memory` (or "forget everything you know about
me") erases it.

## Dispositions

The same crew, a different manner. The factual half of the system prompt —
operator, date, host, spoken-output constraints — is fixed, so switching
personality can never cost the model its grounding.

| Persona | Manner |
| --- | --- |
| `jarvis` | composed British butler-engineer, dryly witty |
| `concise` | at most two sentences, no preamble |
| `engineer` | exact numbers, named trade-offs, stated confidence |
| `socratic` | answers, then asks the question you had not considered |
| `friday` | warmer, faster, more informal |

Say *"be more concise"*, *"engineering mode"*, *"switch persona to Friday"*.

## Instruments

The top bar reports what the assistant actually costs: time to first spoken
word, tokens/second, spike rate, node count, directives handled. Click it for
the full view — p50/p95 latency, a run history where each bar separates *waiting
for the first word* from *generating*, and the last directive's full timing
breakdown including which tools it used.

All of it is measured around real model calls, not estimated.

## Voice control

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
| `be more concise` / `engineering mode` / `switch persona to Friday` | Switches disposition |
| `remember that my sister is called Ada` | Stores a durable fact |
| `what do you know about the deploy script` | Searches every past session |
| `make a note: rewire the proxy` / `list my notes` | Notes |
| `remind me to stretch in 20 minutes` / `list my reminders` | Spoken reminders |
| `what do you remember` | Memory statistics |
| `list skills` / `enable tools` / `disable tools` | The skill kit |
| `performance report` | Measured latency and throughput |
| `louder` / `quieter` / `set volume to 70` / `mute` | Output gain |
| `use the full crew` / `fast mode` | Single pass vs. four-persona pipeline |
| `forget everything you know about me` | Erases the long-term store |

Control phrases are **anchored**: an ordinary question like *"how do I make
this loop run faster?"* goes to the model rather than being swallowed as a
speech-rate command.

### Keyboard

| Key | Action |
| --- | --- |
| `/` | Focus the command bar |
| `⌘P` / `Ctrl+P` | Command palette — every capability, fuzzy-searchable |
| `⌘P` → `theme` | Six colour schemes; the WebGL scene cross-fades with the panels |
| `?` | Reference card — shortcuts, phrases and what is armed right now |
| `Esc` | Barge-in — stop speaking, or close a panel |
| `⌘K` / `Ctrl+K` | Toggle wake-word listening |
| `↑` / `↓` | Command history |

The palette is the readable index of the same surface the voice commands reach:
control phrases, dispositions, installed models, voices, colour schemes and
every registered skill. Skills with no required arguments run directly from it;
everything else dispatches the natural-language directive, so the backend's
intent grammar stays the single source of truth for what a command means.

### Wake word

Click **WAKE** (or `⌘K`) for continuous listening: J.A.R.V.I.S. ignores the
room until it hears *"Jarvis, …"*. An echo guard drops anything the microphone
picks up while the assistant is speaking, so it never answers its own voice —
unless you deliberately say the wake word to interrupt.

### End-to-end check

```bash
make test           # pure-logic checks - no server, no model, no audio device
make smoke          # health, vitals, settings, intents, barge-in, real audio
```

`make test` covers the streamed-speech segmenter, which is the one part of the
speech path that fails *silently*: a bad split does not raise, it just makes
the assistant say "Dr" and then, half a second later, "Stark is in the lab".
The checks feed it the same text at chunk sizes from one character to the whole
string and assert nothing is lost, duplicated, or split inside an abbreviation,
a decimal, or a URL.

`smoke` asserts the audio stream is actually **audible** (non-trivial duration
and peak amplitude) rather than merely present, so a silent regression fails
instead of passing quietly. Add `--wav out.wav` to keep the sample and judge
the voice by ear.

It also checks the parts that are easy to break silently: that the cognitive
graph is fully connected with no orphan nodes, that the file skills genuinely
refuse to escape the workspace or read credentials, that a stored fact survives
a round-trip through recall, and that the reflex arc grounds large arithmetic
exactly while staying silent on questions with nothing to compute. Memory
probes clean up after themselves, so running it does not pollute your store.

## What runs where

| Subsystem | Location | Notes |
| --- | --- | --- |
| Neural shell | `frontend/src/components/NeuralMesh.tsx` | Layered barrel of somas, bezier synapses and travelling action potentials, rebuilt when the topology grows |
| Cognitive graph | `backend/app/neural.py`, `frontend/src/state/neural.ts` | Node/edge model of the running assistant; 20 Hz coalesced activation |
| Cortex map | `frontend/src/components/CortexPanel.tsx` | Region + per-node meters, painted from rAF with zero React renders |
| Skills | `backend/app/skills.py` | Sandboxed tool kit with JSON schemas for native tool calling |
| Reflex arc | `backend/app/reflexes.py` | Deterministic grounding that fires before the cortex |
| Long-term memory | `backend/app/memory.py` | SQLite episodic/semantic store, FTS5 recall, notes, reminders |
| Instrumentation | `backend/app/metrics.py`, `frontend/src/components/Instruments.tsx` | Real timings around real model calls |
| Dispositions | `backend/app/personas.py` | Five personalities over one fixed factual prompt |
| Command palette | `frontend/src/components/CommandPalette.tsx` | ⌘P index of every capability |
| Colour schemes | `frontend/src/state/theme.ts` | Four palettes, shared by the CSS and the WebGL scene |
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
`{"type":"stop"}`, `{"type":"ping","ts":…}`, `{"type":"settings",…}` (a settings
delta) or `{"type":"skill","name":…,"arguments":{…}}`; server sends:

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
| `neural.graph` | The cognitive graph: nodes + edges. Re-sent when a tool node grows |
| `neural` | Coalesced activation at 20 Hz: spikes, edge flows, per-node levels, region peaks |
| `metrics` | Rolling latency / throughput window after every directive |
| `tool` | One executed skill: name, arguments, result, duration |
| `reminder` / `reminder.set` | A reminder fired, or was scheduled |
| `model.pull` | Progress of a model download started from the interface |
| `memory.changed` | Something in the durable store was erased — open archives reload |

`/ws/audio` — server sends `tts.start` (engine, voice, sample_rate, text),
`tts.chunk` (base64 int16 LE PCM, ~200 ms frames), `tts.end`, `tts.flush`
(barge-in) and `tts.error`; client may send `{"type":"speak","text":…}` or
`{"type":"stop"}`. It is a **broadcast** bus — always match on `utterance_id`.

## Configuration (env)

These are **first-boot** defaults. Anything you change from the interface is
saved to `~/.jarvis/settings.json` and wins on every subsequent boot — see
[Settings survive the restart](#settings-survive-the-restart). `DELETE
/api/settings` erases the file and hands control back to the environment.

| Variable | Default | Purpose |
| --- | --- | --- |
| `JARVIS_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `JARVIS_OLLAMA_MODEL` | *(auto-select)* | Crew model |
| `JARVIS_CREW_MODE` | `fast` | `fast` / `crew` |
| `JARVIS_THINK` | `false` | Extended chain-of-thought |
| `JARVIS_MEMORY_TURNS` | `8` | Conversational turns retained in context |
| `JARVIS_PERSONA` | `jarvis` | Disposition preset |
| `JARVIS_TOOLS` | `true` | Let the cortex call skills natively |
| `JARVIS_TOOL_ROUNDS` | `4` | Model→tool→model round-trips per directive |
| `JARVIS_OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama holds the model in memory |
| `JARVIS_PRELOAD` | `true` | Page the model in at boot and on every model switch |
| `JARVIS_NUM_CTX` | `4096` | Context window sent to Ollama (`0` = server default) |
| `JARVIS_NUM_PREDICT` | `512` | Ceiling on generated tokens per answer |
| `JARVIS_OLLAMA_RETRIES` | `2` | Retries for a transient transport failure |
| `JARVIS_RECALL` | `true` | Inject fragments of past sessions |
| `JARVIS_RECALL_LIMIT` | `4` | Fragments injected per directive |
| `JARVIS_PERSIST` | `true` | Write every turn to the on-disk store |
| `JARVIS_DATA_DIR` | `~/.jarvis` | Memory database, saved settings, and the write scratch folder |
| `JARVIS_WORKSPACE` | `~` | Root the file skills may never escape |
| `JARVIS_ALLOW_SHELL` | `false` | Expose the allow-listed `run_command` skill |
| `JARVIS_ALLOW_NET` | `false` | Expose `web_search`, `get_weather` and `fetch_url` |
| `JARVIS_REMINDER_INTERVAL_S` | `20` | How often the reminder scheduler wakes |
| `JARVIS_VOLUME` | `0.9` | Default playback gain |
| `JARVIS_USE_CREWAI` | `false` | Use real CrewAI crew when installed |
| `JARVIS_TTS_ENGINE` | `auto` | `auto` / `kokoro` / `fallback` |
| `JARVIS_TTS_VOICE` | `bm_george` | Kokoro voice id |
| `JARVIS_TTS_DEVICE` | `auto` | ONNX provider: `cpu` / `coreml` / `cuda` / … |
| `JARVIS_TTS_QUALITY` | `fp32` | `fp32` / `q8` / `q4` … |
| `JARVIS_TTS_SPEED` | `1.0` | Speech rate |
| `JARVIS_TTS_WARMUP` | `true` | Pay ONNX session init at boot |
| `JARVIS_STREAM_SPEECH` | `true` | Speak each sentence as it is written |
| `JARVIS_SPEECH_FIRST_MAX_CHARS` | `150` | Longest opening fragment before it breaks at a clause |
| `JARVIS_SPEECH_MAX_CHARS` | `240` | Longest fragment with no sentence end in it |
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
6. **The nervous system** — `neural.py`, `NeuralMesh.tsx`, `CortexPanel.tsx`
7. **Capability** — `skills.py`, `reflexes.py`, native tool calling in `agents/ollama_runtime.py`
8. **Persistence** — `memory.py`, durable recall, notes, reminders
9. **Instrumentation** — `metrics.py`, `Instruments.tsx`, `Overlays.tsx`
