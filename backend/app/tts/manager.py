"""Async TTS manager: bridges blocking engine streams to WebSocket clients."""

from __future__ import annotations

import asyncio
import base64
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

from ..config import settings
from ..logbus import LogBus
from .base import SpeechChunk, TTSEngine


@dataclass
class AudioClient:
    queue: asyncio.Queue
    id: str


class TTSManager:
    """Synthesises utterances and broadcasts PCM frames to /ws/audio clients."""

    def __init__(self, bus: LogBus, engine: TTSEngine) -> None:
        self.bus = bus
        self.engine = engine
        self.clients: dict[str, AudioClient] = {}
        self._current: asyncio.Task | None = None
        self._utterance_seq = 0

    @property
    def client_count(self) -> int:
        return len(self.clients)

    # -- client registry ------------------------------------------------------

    async def subscribe(self) -> AudioClient:
        client = AudioClient(queue=asyncio.Queue(maxsize=256), id=uuid.uuid4().hex[:8])
        self.clients[client.id] = client
        self.bus.publish("info", "audio", f"audio stream attached ({self.client_count} client(s))")
        return client

    async def unsubscribe(self, client: AudioClient) -> None:
        self.clients.pop(client.id, None)
        self.bus.publish("info", "audio", f"audio stream detached ({self.client_count} client(s))")

    def _broadcast(self, message: dict) -> None:
        for client in list(self.clients.values()):
            try:
                client.queue.put_nowait(message)
            except asyncio.QueueFull:
                try:
                    client.queue.get_nowait()
                    client.queue.put_nowait(message)
                except Exception:  # pragma: no cover
                    pass

    # -- synthesis ---------------------------------------------------------------

    async def speak(self, text: str, interrupt: bool = True) -> str:
        """Synthesise `text`, cancelling any utterance already in flight."""
        text = " ".join(text.split())
        if not text:
            return ""
        if interrupt and self._current and not self._current.done():
            self._current.cancel()
            await asyncio.gather(self._current, return_exceptions=True)
            self.bus.publish("voice", "tts", "previous utterance interrupted")
        self._utterance_seq += 1
        utterance_id = f"utt-{self._utterance_seq:04d}"
        task = asyncio.create_task(
            self._run_utterance(utterance_id, text), name=f"tts-{utterance_id}"
        )
        self._current = task
        # Await completion so callers can gate UI status on playback finishing.
        await asyncio.gather(task, return_exceptions=True)
        return utterance_id

    async def _run_utterance(self, utterance_id: str, text: str) -> None:
        started = time.monotonic()
        frames = 0
        samples = 0
        producer: asyncio.Future | None = None
        stop = threading.Event()
        self._broadcast({
            "type": "tts.start",
            "utterance_id": utterance_id,
            "engine": self.engine.name,
            "voice": getattr(self.engine, "voice", "default"),
            "sample_rate": self.engine.sample_rate,
            "text": text,
        })
        self.bus.publish("voice", "tts", f"synthesising {len(text)} chars with {self.engine.name}")
        try:
            queue: asyncio.Queue[SpeechChunk | None] = asyncio.Queue(maxsize=32)
            loop = asyncio.get_running_loop()

            def safe_put(item: SpeechChunk | None) -> None:
                """Thread-safe enqueue that never raises into the loop."""
                try:
                    queue.put_nowait(item)
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()
                        queue.put_nowait(item)
                    except Exception:  # pragma: no cover - defensive
                        pass

            def produce() -> None:
                try:
                    for chunk in self.engine.stream(text):
                        if stop.is_set():
                            return
                        loop.call_soon_threadsafe(safe_put, chunk)
                except Exception as exc:  # noqa: BLE001
                    err = exc
                    loop.call_soon_threadsafe(safe_put, _Errored(err))
                    return
                loop.call_soon_threadsafe(safe_put, None)

            producer = loop.run_in_executor(None, produce)
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, _Errored):
                    raise item.error
                frames += 1
                samples += int(item.pcm.size)
                self._broadcast({
                    "type": "tts.chunk",
                    "utterance_id": utterance_id,
                    "seq": frames,
                    "sample_rate": item.sample_rate,
                    "data": base64.b64encode(item.pcm.tobytes()).decode("ascii"),
                })
                # Yield control so socket sends stay prompt.
                await asyncio.sleep(0)

            duration = samples / max(1, self.engine.sample_rate)
            self._broadcast({
                "type": "tts.end",
                "utterance_id": utterance_id,
                "frames": frames,
                "duration_s": round(duration, 2),
            })
            self.bus.publish(
                "voice",
                "tts",
                f"utterance {utterance_id} complete - {frames} frames, {duration:.1f}s audio",
            )
        except asyncio.CancelledError:
            self._broadcast({"type": "tts.end", "utterance_id": utterance_id, "frames": frames, "cancelled": True})
            self.bus.publish("warn", "tts", f"utterance {utterance_id} cancelled mid-stream")
            raise
        except Exception as exc:  # noqa: BLE001
            self._broadcast({"type": "tts.error", "utterance_id": utterance_id, "detail": str(exc)})
            self.bus.publish("error", "tts", f"synthesis failed: {type(exc).__name__}: {exc}")
        finally:
            stop.set()  # halt the producer thread promptly on cancel/exit
            if producer is not None:
                await asyncio.gather(producer, return_exceptions=True)
            elapsed = time.monotonic() - started
            self.bus.publish("info", "tts", f"engine wall-clock {elapsed:.2f}s")

    async def stop(self) -> None:
        if self._current and not self._current.done():
            self._current.cancel()
            await asyncio.gather(self._current, return_exceptions=True)


@dataclass
class _Errored:
    error: Exception
