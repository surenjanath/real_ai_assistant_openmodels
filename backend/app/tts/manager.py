"""Async TTS manager: bridges blocking engine streams to WebSocket clients."""

from __future__ import annotations

import asyncio
import base64
import threading
import time
import uuid
from dataclasses import dataclass

from ..config import settings
from ..logbus import LogBus
from .base import SpeechChunk, TTSEngine


@dataclass
class AudioClient:
    queue: asyncio.Queue
    id: str


@dataclass
class _Errored:
    error: Exception


class TTSManager:
    """Synthesises utterances and broadcasts PCM frames to /ws/audio clients."""

    def __init__(self, bus: LogBus, engine: TTSEngine) -> None:
        self.bus = bus
        self.engine = engine
        self.clients: dict[str, AudioClient] = {}
        self._current: asyncio.Task | None = None
        self._utterance_seq = 0
        self.last_text: str = ""

    @property
    def client_count(self) -> int:
        return len(self.clients)

    @property
    def speaking(self) -> bool:
        return self._current is not None and not self._current.done()

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
        """Synthesise `text`, cancelling any utterance already in flight.

        Returns once the audio has (approximately) finished *playing*, not
        merely finished synthesising - Kokoro runs ~2x faster than realtime, so
        returning at synthesis-end would flip the interface back to idle while
        the user is still being spoken to.
        """
        text = " ".join(text.split())
        if not text:
            return ""
        if interrupt:
            await self.stop(reason="superseded")
        self._utterance_seq += 1
        utterance_id = f"utt-{self._utterance_seq:04d}"
        self.last_text = text
        task = asyncio.create_task(
            self._run_utterance(utterance_id, text), name=f"tts-{utterance_id}"
        )
        self._current = task
        await asyncio.gather(task, return_exceptions=True)
        return utterance_id

    async def stop(self, reason: str = "interrupted") -> bool:
        """Barge-in: cancel the in-flight utterance and flush client buffers."""
        task = self._current
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._broadcast({"type": "tts.flush", "reason": reason})
        self.bus.publish("warn", "tts", f"utterance {reason}")
        return True

    async def _run_utterance(self, utterance_id: str, text: str) -> None:
        started = time.monotonic()
        first_chunk_at: float | None = None
        frames = 0
        samples = 0
        producer: asyncio.Future | None = None
        stop = threading.Event()
        sample_rate = self.engine.sample_rate

        self._broadcast({
            "type": "tts.start",
            "utterance_id": utterance_id,
            "engine": self.engine.name,
            "voice": getattr(self.engine, "voice", "default"),
            "sample_rate": sample_rate,
            "text": text,
        })
        self.bus.publish("voice", "tts", f"synthesising {len(text)} chars with {self.engine.name}")
        try:
            queue: asyncio.Queue = asyncio.Queue(maxsize=64)
            loop = asyncio.get_running_loop()

            def safe_put(item) -> None:
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
                    loop.call_soon_threadsafe(safe_put, _Errored(exc))
                    return
                loop.call_soon_threadsafe(safe_put, None)

            producer = loop.run_in_executor(None, produce)
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, _Errored):
                    raise item.error
                if not isinstance(item, SpeechChunk) or item.pcm.size == 0:
                    continue
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic()
                    self.bus.publish(
                        "voice", "tts",
                        f"first audio in {first_chunk_at - started:.2f}s",
                    )
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

            duration = samples / max(1, sample_rate)
            self._broadcast({
                "type": "tts.end",
                "utterance_id": utterance_id,
                "frames": frames,
                "duration_s": round(duration, 2),
            })
            synth_elapsed = time.monotonic() - started
            self.bus.publish(
                "voice",
                "tts",
                f"utterance {utterance_id} synthesised - {frames} frames, "
                f"{duration:.1f}s audio in {synth_elapsed:.1f}s (rtf {synth_elapsed / max(0.01, duration):.2f})",
            )
            # Gate on playback: hold until the audio we streamed has had time to
            # play out, so the interface stays in "speaking" for its real span.
            if first_chunk_at is not None:
                remaining = duration - (time.monotonic() - first_chunk_at)
                if remaining > 0:
                    await asyncio.sleep(remaining)
        except asyncio.CancelledError:
            self._broadcast({
                "type": "tts.end", "utterance_id": utterance_id,
                "frames": frames, "cancelled": True,
            })
            raise
        except Exception as exc:  # noqa: BLE001
            self._broadcast({"type": "tts.error", "utterance_id": utterance_id, "detail": str(exc)})
            self.bus.publish("error", "tts", f"synthesis failed: {type(exc).__name__}: {exc}")
        finally:
            stop.set()  # halt the producer thread promptly on cancel/exit
            if producer is not None:
                await asyncio.gather(producer, return_exceptions=True)

    async def shutdown(self) -> None:
        await self.stop(reason="shutdown")
