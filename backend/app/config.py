"""Central configuration for the J.A.R.V.I.S. backend.

Every value can be overridden with a `JARVIS_` prefixed environment variable,
e.g. `JARVIS_TTS_ENGINE=kokoro` or `JARVIS_OLLAMA_MODEL=qwen3:8b`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    version: str = "1.2.0"
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
    tts_frame_samples: int = field(default_factory=lambda: _env_int("TTS_FRAME_SAMPLES", 4800))  # ~200ms @ 24kHz
    tts_max_chars: int = field(default_factory=lambda: _env_int("TTS_MAX_CHARS", 1800))

    # --- Server ------------------------------------------------------------
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "*"))


settings = Settings()
