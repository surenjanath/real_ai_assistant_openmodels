"""J.A.R.V.I.S. backend - FastAPI nervous system (PRD §4).

Endpoints
---------
GET  /api/health        engine/model status snapshot
GET  /api/vitals        latest host metrics sample
GET  /api/settings      current model / voice / speed / persona registry
POST /api/settings      change model / voice / speed / persona / tools / volume
POST /api/command       {text}  - enqueue a command for the crew
POST /api/speak         {text}  - direct TTS (used by n8n / MCP bridges)
POST /api/stop          barge-in: cut off the current utterance
GET  /api/neural        the cognitive graph (nodes + edges) and live activation
GET  /api/metrics       rolling latency / throughput window
GET  /api/skills        the skill catalogue
POST /api/skills/{name} invoke a skill directly (same path the cortex uses)
GET  /api/personas      selectable dispositions
GET  /api/memory        long-term memory statistics
GET  /api/memory/search recall past exchanges by keyword
GET  /api/memory/turns  recent exchanges
DELETE /api/memory      erase the long-term store
GET/POST /api/notes     list / create notes
DELETE /api/notes/{id}  delete a note
GET/POST /api/reminders  list / create reminders
DELETE /api/reminders/{id}  cancel a reminder
GET  /api/facts         durable facts the assistant has been told
DELETE /api/facts/{key} forget one fact
GET  /api/export        the conversation as text or JSON
WS   /ws/logs           bidirectional telemetry: server pushes log + status +
                        vitals + neural + metrics + answer frames, client
                        pushes commands / STT
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
from fastapi.responses import PlainTextResponse

from . import __version__
from .config import settings
from .logbus import LogBus, LogEvent
from .memory import Memory
from .metrics import Metrics
from .neural import NeuralBus
from .orchestrator import Orchestrator
from .personas import catalogue as persona_catalogue
from .registry import Registry
from .skills import SkillKit, workspace_root
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
_neural = NeuralBus(_bus)
_metrics = Metrics(_bus)
_vitals = Vitals(_bus, client_provider=lambda: len(_bus.subscribers) + _tts.client_count)

try:
    _memory: Memory | None = Memory()
except Exception as exc:  # noqa: BLE001 - a broken store must not stop the boot
    _memory = None
    print(f"[   warn] memory        store unavailable: {type(exc).__name__}: {exc}", flush=True)

_kit = SkillKit(_memory, vitals_provider=lambda: _vitals.latest) if _memory else None
_orchestrator = Orchestrator(
    _bus, _tts, _registry, memory=_memory, kit=_kit, neural=_neural, metrics=_metrics,
    vitals_provider=lambda: _vitals.latest,
)
_telemetry = TelemetrySimulator(
    _bus,
    _vitals,
    client_provider=lambda: len(_bus.subscribers) + _tts.client_count,
)


def _engine_label() -> str:
    return getattr(_engine, "label", _engine.name)


def _engines_frame() -> dict[str, Any]:
    return {
        "tts": _engine.name,
        "tts_label": _engine_label(),
        "tts_mode": _engine_mode,
        "agents": _orchestrator.runtime_name,
        "mode": _orchestrator.mode,
        "model": _registry.model,
    }


async def _autonomic() -> None:
    """Idle-time housekeeping: keep the host-sense node alive on the graph and
    refresh the Ollama catalogue so a model pulled while running shows up."""
    while True:
        await asyncio.sleep(20.0)
        try:
            _neural.fire("sense.host", 0.35)
            await _registry.refresh_models()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _bus.bind_loop()
    await _neural.start()
    await _vitals.start()
    await _telemetry.start()
    await _orchestrator.start()
    housekeeping = asyncio.create_task(_autonomic(), name="autonomic")
    if _memory is not None:
        stats = _memory.stats()
        _bus.publish(
            "success", "memory",
            f"hippocampus online - {stats['turns']} exchanges, {stats['facts']} facts "
            f"({'fts5' if stats['fts'] else 'scan'} recall)",
        )
    if _kit is not None:
        gated = [s.name for s in _kit.skills.values() if s.danger != "safe"]
        _bus.publish(
            "info", "skills",
            f"{len(_kit.skills)} skills registered"
            + (f" ({len(gated)} privileged)" if gated else "")
            + f" - workspace {workspace_root()}",
        )
    # Pay the ONNX session-init cost now so the first real answer is not slow.
    if settings.tts_warmup and hasattr(_engine, "warmup"):
        async def _warm() -> None:
            started = time.monotonic()
            await asyncio.to_thread(_engine.warmup)
            _bus.publish("success", "tts", f"vocal engine warm in {time.monotonic() - started:.1f}s")
        asyncio.create_task(_warm())
    yield
    housekeeping.cancel()
    await asyncio.gather(housekeeping, return_exceptions=True)
    await _telemetry.stop()
    await _vitals.stop()
    await _neural.stop()
    await _orchestrator.stop()
    await _tts.shutdown()
    if _memory is not None:
        _memory.close()
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
                "persona": _registry.persona,
                "tools": _registry.tools_active,
            },
            "memory": {
                "available": _memory is not None,
                "recall": _registry.recall,
            },
            "neural": {"nodes": len(_neural.nodes), "edges": len(_neural.edges),
                       "fired": _neural.fired},
        },
        "skills": len(_kit.skills) if _kit else 0,
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
    """Change model / voice / speed / persona / tools from the panel (or curl)."""
    return _registry.apply(
        model=payload.get("model"),
        voice=payload.get("voice"),
        speed=payload.get("speed"),
        think=payload.get("think"),
        persona=payload.get("persona"),
        tools=payload.get("tools"),
        recall=payload.get("recall"),
        volume=payload.get("volume"),
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


# ------------------------------------------------------------- cognition ---


@app.get("/api/neural")
async def neural_graph() -> dict[str, Any]:
    frame = _neural.graph_frame()
    frame["levels"] = [round(v, 3) for v in _neural.activation]
    frame["fired"] = _neural.fired
    return frame


@app.get("/api/metrics")
async def metrics() -> dict[str, Any]:
    return _metrics.snapshot()


@app.get("/api/personas")
async def personas() -> dict[str, Any]:
    return {"ok": True, "current": _registry.persona, "personas": persona_catalogue()}


@app.get("/api/skills")
async def skills() -> dict[str, Any]:
    if _kit is None:
        return {"ok": False, "error": "skill kit unavailable", "skills": []}
    return {
        "ok": True,
        "armed": _registry.tools_active,
        "supported": _registry.tools_supported,
        "calls": _kit.calls,
        "workspace": str(workspace_root()),
        "shell": _kit.allow_shell,
        "network": _kit.allow_net,
        "skills": _kit.catalogue(),
    }


@app.post("/api/skills/{name}")
async def invoke_skill(name: str, payload: dict | None = None) -> dict[str, Any]:
    """Run a skill by hand — the same code path the cortex takes, so the panel
    can be used to check a tool works before trusting a model with it."""
    if _kit is None:
        return {"ok": False, "error": "skill kit unavailable"}
    node = _neural.ensure_tool_node(name)
    _neural.signal("motor.tools", node, 1.0)
    result = await asyncio.to_thread(_kit.invoke, name, (payload or {}).get("arguments") or {})
    _bus.push_frame({
        "type": "tool", "name": name, "args": (payload or {}).get("arguments") or {},
        "ok": bool(result.get("ok")),
        "detail": str(result.get("error") or result.get("result"))[:400],
        "elapsed_ms": result.get("elapsed_ms", 0), "ts": time.time(),
    })
    return result


# ---------------------------------------------------------------- memory ---


def _no_memory() -> dict[str, Any]:
    return {"ok": False, "error": "long-term memory unavailable"}


@app.get("/api/memory")
async def memory_stats() -> dict[str, Any]:
    if _memory is None:
        return _no_memory()
    return {"ok": True, "stats": await asyncio.to_thread(_memory.stats)}


@app.get("/api/memory/search")
async def memory_search(q: str = "", limit: int = 12) -> dict[str, Any]:
    if _memory is None:
        return _no_memory()
    if not q.strip():
        return {"ok": False, "error": "q required"}
    hits = await asyncio.to_thread(_memory.search, q, max(1, min(50, limit)))
    return {"ok": True, "query": q, "matches": [h.as_dict() for h in hits]}


@app.get("/api/memory/turns")
async def memory_turns(limit: int = 40, session_only: bool = False) -> dict[str, Any]:
    if _memory is None:
        return _no_memory()
    turns = await asyncio.to_thread(_memory.recent_turns, max(1, min(300, limit)), session_only)
    return {"ok": True, "turns": turns}


@app.delete("/api/memory")
async def memory_wipe() -> dict[str, Any]:
    if _memory is None:
        return _no_memory()
    removed = await asyncio.to_thread(_memory.forget_all)
    _bus.publish("warn", "memory", f"long-term memory erased via API - {removed} turn(s)")
    return {"ok": True, "removed": removed}


@app.get("/api/facts")
async def facts() -> dict[str, Any]:
    if _memory is None:
        return _no_memory()
    return {"ok": True, "facts": await asyncio.to_thread(_memory.all_facts, 100)}


@app.delete("/api/facts/{key}")
async def forget_fact(key: str) -> dict[str, Any]:
    if _memory is None:
        return _no_memory()
    return {"ok": await asyncio.to_thread(_memory.forget_fact, key)}


@app.get("/api/notes")
async def list_notes(limit: int = 50) -> dict[str, Any]:
    if _memory is None:
        return _no_memory()
    return {"ok": True, "notes": await asyncio.to_thread(_memory.list_notes, limit)}


@app.post("/api/notes")
async def add_note(payload: dict) -> dict[str, Any]:
    if _memory is None:
        return _no_memory()
    text = str(payload.get("text", "")).strip()
    if not text:
        return {"ok": False, "error": "text required"}
    note_id = await asyncio.to_thread(_memory.add_note, text, str(payload.get("tags", "")))
    _bus.publish("success", "notes", f"#{note_id} {text[:120]}")
    return {"ok": True, "id": note_id}


@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: int) -> dict[str, Any]:
    if _memory is None:
        return _no_memory()
    return {"ok": await asyncio.to_thread(_memory.delete_note, note_id)}


@app.get("/api/reminders")
async def list_reminders() -> dict[str, Any]:
    if _memory is None:
        return _no_memory()
    return {"ok": True, "reminders": await asyncio.to_thread(_memory.pending_reminders, 50)}


@app.post("/api/reminders")
async def add_reminder(payload: dict) -> dict[str, Any]:
    if _memory is None or _kit is None:
        return _no_memory()
    text = str(payload.get("text", "")).strip()
    when = str(payload.get("when", "")).strip()
    if not text or not when:
        return {"ok": False, "error": "text and when required"}
    result = await asyncio.to_thread(_kit.invoke, "set_reminder", {"text": text, "when": when})
    return result


@app.delete("/api/reminders/{reminder_id}")
async def cancel_reminder(reminder_id: int) -> dict[str, Any]:
    if _memory is None:
        return _no_memory()
    return {"ok": await asyncio.to_thread(_memory.cancel_reminder, reminder_id)}


@app.get("/api/export")
async def export(fmt: str = "text", limit: int = 200) -> Any:
    """Hand the conversation back to the operator — it is their data."""
    if _memory is None:
        return _no_memory()
    turns = await asyncio.to_thread(_memory.recent_turns, max(1, min(2000, limit)), False)
    if fmt == "json":
        return {"ok": True, "turns": turns}
    lines = []
    for turn in turns:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(turn["ts"]))
        who = "YOU" if turn["role"] == "user" else settings.name
        lines.append(f"[{stamp}] {who}: {turn['text']}")
    return PlainTextResponse("\n".join(lines) or "(no exchanges recorded)")


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
            "engines": _engines_frame(),
            "settings": _registry.as_dict(),
            "vitals": _vitals.latest,
            "skills": _kit.catalogue() if _kit else [],
            "memory": (await asyncio.to_thread(_memory.stats)) if _memory else None,
        })
        # The cognitive graph before any backlog, so spike frames replayed
        # below always land on a client that already knows the topology.
        await ws.send_json(_neural.graph_frame())
        await ws.send_json(_metrics.snapshot())
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
        elif mtype == "settings":
            _registry.apply(
                model=message.get("model"), voice=message.get("voice"),
                speed=message.get("speed"), think=message.get("think"),
                persona=message.get("persona"), tools=message.get("tools"),
                recall=message.get("recall"), volume=message.get("volume"),
            )
        elif mtype == "skill":
            name = str(message.get("name", "")).strip()
            if name and _kit is not None:
                asyncio.create_task(invoke_skill(name, {"arguments": message.get("arguments") or {}}))
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
