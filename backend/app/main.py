"""J.A.R.V.I.S. backend - FastAPI nervous system (PRD §4).

Endpoints
---------
GET  /api/health        engine/model status snapshot
GET  /api/vitals        latest host metrics sample
GET  /api/settings      current model / voice / speed registry
POST /api/settings      change model / voice / speed
POST /api/command       {text}  - enqueue a command for the crew
POST /api/speak         {text}  - direct TTS (used by n8n / MCP bridges)
POST /api/stop          barge-in: cut off the current utterance
WS   /ws/logs           bidirectional telemetry: server pushes log + status +
                        vitals + answer frames, client pushes commands / STT
WS   /ws/audio          server pushes base64 int16 PCM TTS chunks; client can
                        push {"type":"speak","text":...}
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import settings
from .logbus import LogBus, LogEvent
from .orchestrator import Orchestrator
from .registry import Registry
from .telemetry import TelemetrySimulator
from .tts import build_engine
from .tts.manager import TTSManager
from .vitals import Vitals

_started = time.monotonic()
_bus = LogBus(backlog_size=settings.log_backlog)
_engine, _engine_mode = build_engine(_bus)
_tts = TTSManager(_bus, _engine)
_registry = Registry(bus=_bus)
_registry.attach_engine(_engine)
_orchestrator = Orchestrator(_bus, _tts, _registry)
_vitals = Vitals(_bus, client_provider=lambda: len(_bus.subscribers) + _tts.client_count)
_telemetry = TelemetrySimulator(
    _bus,
    _vitals,
    client_provider=lambda: len(_bus.subscribers) + _tts.client_count,
)


def _engine_label() -> str:
    return getattr(_engine, "label", _engine.name)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _bus.bind_loop()
    await _vitals.start()
    await _telemetry.start()
    await _orchestrator.start()
    # Pay the ONNX session-init cost now so the first real answer is not slow.
    if settings.tts_warmup and hasattr(_engine, "warmup"):
        async def _warm() -> None:
            started = time.monotonic()
            await asyncio.to_thread(_engine.warmup)
            _bus.publish("success", "tts", f"vocal engine warm in {time.monotonic() - started:.1f}s")
        asyncio.create_task(_warm())
    yield
    await _telemetry.stop()
    await _vitals.stop()
    await _orchestrator.stop()
    await _tts.shutdown()
    _bus.publish("warn", "shutdown", "backend going down")


app = FastAPI(title=settings.name, version=__version__, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- REST -----


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "name": settings.name,
        "version": __version__,
        "uptime_s": round(time.monotonic() - _started, 1),
        "engines": {
            "tts": {
                "name": _engine.name,
                "label": _engine_label(),
                "mode": _engine_mode,
                "voice": _registry.voice,
                "sample_rate": _engine.sample_rate,
            },
            "agents": {
                "name": _orchestrator.runtime_name,
                "mode": _orchestrator.mode,
                "model": _registry.model,
                "verified": _registry.model_verified,
            },
        },
        "clients": {"logs": len(_bus.subscribers), "audio": _tts.client_count},
        "speaking": _tts.speaking,
    }


@app.get("/api/vitals")
async def vitals() -> dict[str, Any]:
    return _vitals.latest or _vitals.sample()


@app.get("/api/settings")
async def get_settings() -> dict[str, Any]:
    return {
        "ok": True,
        "settings": _registry.as_dict(),
        "engines": {
            "tts": _engine.name,
            "tts_label": _engine_label(),
            "tts_mode": _engine_mode,
            "agents": _orchestrator.runtime_name,
            "mode": _orchestrator.mode,
        },
    }


@app.post("/api/settings")
async def post_settings(payload: dict) -> dict[str, Any]:
    """Change model / voice / speed from the settings panel (or curl)."""
    return _registry.apply(
        model=payload.get("model"),
        voice=payload.get("voice"),
        speed=payload.get("speed"),
        think=payload.get("think"),
    )


@app.post("/api/command")
async def command(payload: dict) -> dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    if not text:
        return {"ok": False, "error": "text required"}
    await _orchestrator.enqueue(text, origin=str(payload.get("origin", "text")))
    return {"ok": True, "queued": text}


@app.post("/api/speak")
async def speak(payload: dict) -> dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    if not text:
        return {"ok": False, "error": "text required"}
    utterance_id = await _tts.speak(text)
    return {"ok": True, "utterance_id": utterance_id, "engine": _engine.name}


@app.post("/api/stop")
async def stop_speaking() -> dict[str, Any]:
    stopped = await _tts.stop(reason="barge-in")
    _bus.push_frame({"type": "status", "status": "idle", "detail": ""})
    return {"ok": True, "stopped": stopped}


# ------------------------------------------------------------ websockets ---


def _serialize(item: Any) -> dict | None:
    if isinstance(item, LogEvent):
        return item.as_dict()
    if isinstance(item, dict):
        return item
    return None


@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket) -> None:
    await ws.accept()
    queue = await _bus.subscribe()
    try:
        await ws.send_json({
            "type": "hello",
            "name": settings.name,
            "version": __version__,
            "operator": settings.operator,
            "engines": {
                "tts": _engine.name,
                "tts_label": _engine_label(),
                "tts_mode": _engine_mode,
                "agents": _orchestrator.runtime_name,
                "mode": _orchestrator.mode,
                "model": _registry.model,
            },
            "settings": _registry.as_dict(),
            "vitals": _vitals.latest,
        })
        for event in list(_bus.backlog)[-settings.log_backlog:]:
            await ws.send_json(event.as_dict())

        receiver = asyncio.create_task(_logs_receiver(ws))
        try:
            while True:
                getter = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {getter, receiver}, return_when=asyncio.FIRST_COMPLETED
                )
                if receiver in done:
                    getter.cancel()
                    break
                item = getter.result()
                if item is None:
                    break
                frame = _serialize(item)
                if frame:
                    await ws.send_json(frame)
        finally:
            receiver.cancel()
            with contextlib.suppress(BaseException):
                await receiver
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        _bus.publish("warn", "ws", f"logs socket error: {type(exc).__name__}")
    finally:
        await _bus.unsubscribe(queue)


async def _logs_receiver(ws: WebSocket) -> None:
    """Inbound half of the telemetry socket: commands and STT transcripts."""
    while True:
        message = await ws.receive_json()
        mtype = message.get("type", "command")
        if mtype == "command":
            text = str(message.get("text", "")).strip()
            origin = str(message.get("origin", "text"))
            if text:
                await _orchestrator.enqueue(text, origin)
        elif mtype == "ping":
            await ws.send_json({"type": "pong", "ts": message.get("ts", time.time())})
        elif mtype == "stop":
            # Barge-in must bypass the command queue entirely.
            asyncio.create_task(_orchestrator._handle_stop())
        elif mtype == "speak":
            text = str(message.get("text", "")).strip()
            if text:
                # Detached so a disconnect cannot cancel the utterance for
                # every other connected client.
                asyncio.create_task(_tts.speak(text))


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket) -> None:
    await ws.accept()
    client = await _tts.subscribe()
    receiver = asyncio.create_task(_audio_receiver(ws))
    try:
        while True:
            getter = asyncio.create_task(client.queue.get())
            done, _ = await asyncio.wait({getter, receiver}, return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                getter.cancel()
                break
            await ws.send_json(getter.result())
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        _bus.publish("warn", "ws", f"audio socket error: {type(exc).__name__}")
    finally:
        receiver.cancel()
        with contextlib.suppress(BaseException):
            await receiver
        await _tts.unsubscribe(client)


async def _audio_receiver(ws: WebSocket) -> None:
    while True:
        message = await ws.receive_json()
        mtype = message.get("type")
        if mtype == "speak":
            text = str(message.get("text", "")).strip()
            if text:
                # Detached: the utterance is broadcast to every audio client,
                # so it must survive this particular socket disconnecting.
                asyncio.create_task(_tts.speak(text))
        elif mtype == "stop":
            asyncio.create_task(_tts.stop(reason="barge-in"))
