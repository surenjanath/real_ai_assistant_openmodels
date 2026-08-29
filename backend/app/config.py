"""Central configuration for the J.A.R.V.I.S. backend.

Every value can be overridden with a `JARVIS_` prefixed environment variable,
e.g. `JARVIS_TTS_ENGINE=kokoro` or `JARVIS_OLLAMA_MODEL=qwen3:8b`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import __version__


def _env(key: str, default: str) -> str:
    value = os.environ.get(f"JARVIS_{key}")
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(f"JARVIS_{key}", default))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(f"JARVIS_{key}", default))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(f"JARVIS_{key}")
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(kw_only=True)
class Settings:
    """Runtime settings, loaded once at startup."""

    # --- Identity / telemetry -------------------------------------------
    name: str = field(default_factory=lambda: _env("NAME", "J.A.R.V.I.S."))
    #: mirrors `app.__version__`; never define a second copy here
    version: str = __version__
    operator: str = field(default_factory=lambda: _env("OPERATOR", "sir"))
    log_backlog: int = field(default_factory=lambda: _env_int("LOG_BACKLOG", 60))
    heartbeat_min_s: float = field(default_factory=lambda: _env_int("HEARTBEAT_MIN_S", 6))
    heartbeat_max_s: float = field(default_factory=lambda: _env_int("HEARTBEAT_MAX_S", 14))
    vitals_interval_s: float = field(default_factory=lambda: _env_float("VITALS_INTERVAL_S", 2.0))

    # --- Agentic layer (Phase 5) -----------------------------------------
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", ""))
    use_crewai: bool = field(default_factory=lambda: _env_bool("USE_CREWAI", False))
    agent_step_timeout_s: float = field(default_factory=lambda: _env_int("AGENT_STEP_TIMEOUT_S", 120))
    # "fast" = single synthesiser pass (conversational, ~1 model call).
    # "crew" = full Router -> Analyst -> Engineer -> Synthesiser pipeline.
    crew_mode: str = field(default_factory=lambda: _env("CREW_MODE", "fast"))
    # Chain-of-thought on reasoning models is OFF by default: it can add tens of
    # seconds before the first spoken word, which ruins a conversational
    # assistant. Users opt in per-session from the panel or by voice.
    think: bool = field(default_factory=lambda: _env_bool("THINK", False))
    memory_turns: int = field(default_factory=lambda: _env_int("MEMORY_TURNS", 8))
    #: Start a fresh working context when nothing has been said for this long.
    #: A conversation left open overnight otherwise arrives tomorrow still
    #: carrying yesterday, and every turn pays to re-read it. 0 disables.
    context_idle_reset_s: float = field(
        default_factory=lambda: _env_float("CONTEXT_IDLE_RESET_S", 900.0)
    )
    #: Headroom kept free for the answer and the per-turn grounding (persona,
    #: reflexes, recall) when deciding how much conversation still fits.
    context_reserve_tokens: int = field(
        default_factory=lambda: _env_int("CONTEXT_RESERVE_TOKENS", 900)
    )
    #: personality preset key - see personas.py
    persona: str = field(default_factory=lambda: _env("PERSONA", "jarvis"))
    #: let the cortex call skills natively when the model supports it
    tools: bool = field(default_factory=lambda: _env_bool("TOOLS", True))
    #: how many tool round-trips one directive may spend
    tool_rounds: int = field(default_factory=lambda: _env_int("TOOL_ROUNDS", 4))
    #: how long Ollama keeps the model resident after a request. Reloading a
    #: 8B model costs 2-10s, which is the single largest source of a slow
    #: first answer; "30m" keeps it warm through a whole working session.
    ollama_keep_alive: str = field(default_factory=lambda: _env("OLLAMA_KEEP_ALIVE", "30m"))
    #: preload the model into VRAM at boot / on model switch, so the first
    #: directive never pays the load cost
    preload: bool = field(default_factory=lambda: _env_bool("PRELOAD", True))
    #: Context window handed to Ollama; 0 leaves the server default alone.
    #:
    #: 8192 rather than 4096 because tool schemas are not free: the 33 skills
    #: this assistant registers cost ~3,200 prompt tokens on their own, before
    #: the persona, the reflex grounding, the recalled fragments or the
    #: conversation. At 4096 that left too little room, and the failure is
    #: silent - Ollama drops the oldest messages to make the prompt fit, so
    #: the assistant forgets the conversation rather than saying it cannot
    #: hold it. `_warn_if_squeezed` now reports when this is close.
    num_ctx: int = field(default_factory=lambda: _env_int("NUM_CTX", 8192))
    #: hard ceiling on generated tokens - a runaway answer is a hung assistant
    num_predict: int = field(default_factory=lambda: _env_int("NUM_PREDICT", 512))
    #: retries for a transient Ollama transport failure
    ollama_retries: int = field(default_factory=lambda: _env_int("OLLAMA_RETRIES", 2))

    # --- Durable memory (Phase 6) -----------------------------------------
    #: inject relevant fragments of past sessions into the prompt
    recall: bool = field(default_factory=lambda: _env_bool("RECALL", True))
    #: how many recalled fragments may be injected per directive
    recall_limit: int = field(default_factory=lambda: _env_int("RECALL_LIMIT", 4))
    #: minimum relevance a fragment needs before it is worth the tokens
    recall_threshold: float = field(default_factory=lambda: _env_float("RECALL_THRESHOLD", 0.18))
    #: persist every turn to the on-disk hippocampus
    persist: bool = field(default_factory=lambda: _env_bool("PERSIST", True))
    #: how often the reminder scheduler wakes, in seconds
    reminder_interval_s: float = field(default_factory=lambda: _env_float("REMINDER_INTERVAL_S", 20.0))

    # --- Vocal engine (Phase 3) ------------------------------------------
    # auto -> try pykokoro, then kokoro, then the dependency-free fallback synth.
    tts_engine: str = field(default_factory=lambda: _env("TTS_ENGINE", "auto"))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "bm_george"))
    tts_speed: float = field(default_factory=lambda: _env_float("TTS_SPEED", 1.0))
    # ONNX execution provider for pykokoro: auto|cpu|coreml|cuda|openvino|...
    tts_device: str = field(default_factory=lambda: _env("TTS_DEVICE", "auto"))
    # Model weights precision. fp32 measures ~2.4x faster than q8 on Apple
    # Silicon (RTF 0.43 vs 1.04) at the cost of a larger one-time download.
    tts_quality: str = field(default_factory=lambda: _env("TTS_QUALITY", "fp32"))
    tts_warmup: bool = field(default_factory=lambda: _env_bool("TTS_WARMUP", True))
    sample_rate: int = 24000  # Kokoro native output rate; fallback matches it.
    #: PCM transport frame size. 100ms rather than 200ms: the first packet of
    #: an utterance leaves the moment it exists instead of waiting to fill a
    #: bigger buffer, and the browser gets finer scheduling granularity.
    tts_frame_samples: int = field(default_factory=lambda: _env_int("TTS_FRAME_SAMPLES", 2400))  # ~100ms @ 24kHz
    tts_max_chars: int = field(default_factory=lambda: _env_int("TTS_MAX_CHARS", 1800))
    #: default interface playback gain, 0..1
    volume: float = field(default_factory=lambda: _env_float("VOLUME", 0.9))
    #: speak each sentence as the model finishes it, rather than waiting for
    #: the whole answer. Cuts time-to-first-word to roughly one sentence of
    #: generation instead of the full completion.
    stream_speech: bool = field(default_factory=lambda: _env_bool("STREAM_SPEECH", True))
    #: shortest fragment worth handing to the vocal engine on its own
    speech_min_chars: int = field(default_factory=lambda: _env_int("SPEECH_MIN_CHARS", 12))
    #: force a break at a soft boundary once a fragment grows past this
    speech_max_chars: int = field(default_factory=lambda: _env_int("SPEECH_MAX_CHARS", 240))
    #: The *first* fragment is allowed to break at a clause boundary (a comma,
    #: a dash) rather than waiting for a full stop. Kokoro only emits audio at
    #: the end of a whole sentence, so with 35-word sentences the opening
    #: fragment alone can cost several seconds of silence. Breaking it early
    #: gets the assistant talking; by the time that clause has been spoken the
    #: rest of the answer is long since synthesised.
    #: Both are measured, not guessed, and the floor is not there for prosody.
    #: Kokoro's time to its first sample against fragment length has a cliff
    #: just under ~32 characters (median over six sentences, this machine):
    #:
    #:     26ch -> 1.15s for 1.5s of speech      38ch -> 0.67s for 3.3s
    #:     30ch -> 1.21s for 1.9s of speech      46ch -> 0.76s for 3.8s
    #:     34ch -> 0.67s for 3.0s of speech      56ch -> 0.91s for 4.5s
    #:
    #: A shorter opening fragment is therefore *slower to speak and shorter
    #: when spoken* - the worst of both - so asking for less than ~34
    #: characters costs half a second and buys nothing. Above the cliff the
    #: cost is roughly linear, and 34-46 is where latency is lowest while
    #: still returning several seconds of audio to synthesise the rest behind.
    speech_first_min_chars: int = field(default_factory=lambda: _env_int("SPEECH_FIRST_MIN_CHARS", 34))
    #: ...and must break by here even if no clause boundary ever turns up. Kept
    #: tight on purpose: an answer with no comma in it at all is entirely
    #: ordinary, and a generous ceiling means those answers are never streamed.
    speech_first_max_chars: int = field(default_factory=lambda: _env_int("SPEECH_FIRST_MAX_CHARS", 46))

    # --- Full-duplex conversation ------------------------------------------
    #: A spoken directive that arrives while J.A.R.V.I.S. is still talking
    #: cuts the utterance off instead of queueing behind it. Without this the
    #: operator waits out the whole answer before the next one even starts.
    barge_in: bool = field(default_factory=lambda: _env_bool("BARGE_IN", True))
    #: How far back a spoken directive is compared against what was just
    #: said aloud. Covers the recogniser's own finalisation lag, which is why
    #: it is seconds rather than milliseconds.
    echo_guard_ms: int = field(default_factory=lambda: _env_int("ECHO_GUARD_MS", 4000))
    #: Fraction of a transcript's content words that must have just been
    #: spoken before it is dismissed as the microphone hearing the speakers.
    echo_similarity: float = field(default_factory=lambda: _env_float("ECHO_SIMILARITY", 0.6))

    # --- Server ------------------------------------------------------------
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "*"))


settings = Settings()
