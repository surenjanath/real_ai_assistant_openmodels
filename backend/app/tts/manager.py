"""Async TTS manager: bridges blocking engine streams to WebSocket clients.

Two ways in:

* :meth:`TTSManager.speak` - synthesise a finished string. Used by the control
  intents, the reminder scheduler and the ``/api/speak`` bridge.
* :meth:`TTSManager.open_stream` - open a :class:`SpeechStream` and feed it the
  model's output as it is generated. Each completed sentence is synthesised
  while the next is still being written, so the assistant starts speaking after
  roughly one sentence of generation instead of the whole answer. On a 90-word
  reply that is the difference between talking at ~4 s and talking at ~1 s.

Both paths share one utterance slot: a new utterance (or an explicit barge-in)
cancels whatever was in flight, and the clients are told to flush.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable

from ..config import settings
from ..logbus import LogBus
from .base import SpeechChunk, TTSEngine
from .segmenter import SentenceSegmenter


@dataclass
class AudioClient:
    queue: asyncio.Queue
    id: str
    #: latched once this client has fallen far enough behind to lose frames,
    #: so the failure is reported once rather than per dropped frame
    overflowed: bool = False


@dataclass
class _Errored:
    error: Exception


class SpeechStream:
    """An utterance assembled from a live token stream.

    `feed` is safe to call from a worker thread - the model's delta callback
    runs inside ``asyncio.to_thread`` - and hands complete sentences to the
    synthesiser as they close.
    """

    def __init__(self, manager: "TTSManager", utterance_id: str,
                 loop: asyncio.AbstractEventLoop) -> None:
        self._manager = manager
        self._loop = loop
        self.utterance_id = utterance_id
        self._segmenter = SentenceSegmenter()
        self._segments: asyncio.Queue = asyncio.Queue()
        self._lock = threading.Lock()
        self._closed = False
        #: set when a barge-in (or a superseding utterance) killed this stream.
        #: The orchestrator must not then "helpfully" speak the answer the
        #: operator just interrupted.
        self.stopped = False
        #: everything actually handed to the vocal engine, for the record
        self.spoken_text = ""
        #: generation number - bumped by abort() so late deltas are ignored
        self._epoch = 0

    # -- producer side (may run off-loop) ------------------------------------

    def feed(self, delta: str) -> None:
        if not delta:
            return
        with self._lock:
            if self._closed:
                return
            fragments = self._segmenter.push(delta)
            epoch = self._epoch
        for fragment in fragments:
            self._submit(fragment, epoch)

    def _submit(self, fragment: str, epoch: int) -> None:
        if threading.current_thread() is threading.main_thread():
            self._enqueue(fragment, epoch)
        else:
            self._loop.call_soon_threadsafe(self._enqueue, fragment, epoch)

    def _enqueue(self, fragment: str, epoch: int, final: bool = False) -> None:
        """Queue one fragment for synthesis.

        `final` is set only by `finish`, which flushes the tail *after* closing
        the stream to further input. Without it the closed-check below would
        reject that flush and the last sentence of every single answer would be
        dropped on the floor - silently, since nothing errors.
        """
        if epoch != self._epoch:
            return
        # A tool round (or a barge-in) invalidated everything queued before it.
        if self._closed and not final:
            return
        self.spoken_text += (" " if self.spoken_text else "") + fragment
        self._segments.put_nowait(fragment)

    # -- consumer side (loop only) -------------------------------------------

    @property
    def started(self) -> bool:
        """True once real audio has left the vocal engine for this utterance."""
        return self._manager.stream_audio_started

    def abort(self) -> None:
        """Discard everything buffered and queued - a tool round intervened.

        Cheap and idempotent. If audio had already gone out, the clients are
        told to flush so the operator does not hear the discarded scaffolding.
        """
        with self._lock:
            self._epoch += 1
            self._segmenter.reset()
        self.spoken_text = ""
        drained = 0
        while not self._segments.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._segments.get_nowait()
                drained += 1
        self._manager.restart_stream_audio()

    async def finish(self) -> bool:
        """Flush the tail, wait for playback, and report whether we spoke.

        A `False` return is the caller's cue to speak the answer the ordinary
        way. Note that a stream killed by a barge-in returns `True`: nothing
        was said, but nothing *should* be said.
        """
        with self._lock:
            if self.stopped:
                return True
            if self._closed:
                return self._manager.stream_audio_started
            self._closed = True
            tail = self._segmenter.drain()
            epoch = self._epoch
        for fragment in tail:
            self._enqueue(fragment, epoch, final=True)
        self._segments.put_nowait(None)
        return await self._manager.await_stream()

    async def cancel(self) -> None:
        with self._lock:
            self._closed = True
            self.stopped = True
        await self._manager.stop(reason="superseded")


class TTSManager:
    """Synthesises utterances and broadcasts PCM frames to /ws/audio clients."""

    def __init__(self, bus: LogBus, engine: TTSEngine) -> None:
        self.bus = bus
        self.engine = engine
        self.clients: dict[str, AudioClient] = {}
        self._current: asyncio.Task | None = None
        self._utterance_seq = 0
        self.last_text: str = ""
        #: monotonic time at which everything streamed so far finishes playing
        self._play_deadline: float | None = None
        #: (playback-end, text) for everything recently spoken aloud, so the
        #: echo guard can ask "did we just say this?" of an open microphone.
        #: Keyed on when each fragment stops *sounding*, not when it was
        #: synthesised - that is the instant the room goes quiet again.
        self._spoken: deque[tuple[float, str]] = deque(maxlen=24)
        #: whether the open stream has emitted any audio at all
        self.stream_audio_started = False
        self._stream: SpeechStream | None = None

    @property
    def client_count(self) -> int:
        return len(self.clients)

    @property
    def speaking(self) -> bool:
        return self._current is not None and not self._current.done()

    # -- client registry ------------------------------------------------------

    #: Frames one client may fall behind before we admit defeat.
    #:
    #: Sized in *seconds of audio*, not frames, because the frame size is a
    #: tunable: at 100ms frames the old 256-frame bound held only 25 seconds,
    #: and answers here routinely run past 30 - so a long reply silently
    #: overflowed and punched holes in itself. 3,000 frames is five minutes at
    #: any frame size this project uses.
    CLIENT_QUEUE_FRAMES = 3000

    async def subscribe(self) -> AudioClient:
        client = AudioClient(
            queue=asyncio.Queue(maxsize=self.CLIENT_QUEUE_FRAMES), id=uuid.uuid4().hex[:8]
        )
        self.clients[client.id] = client
        self.bus.publish("info", "audio", f"audio stream attached ({self.client_count} client(s))")
        return client

    async def unsubscribe(self, client: AudioClient) -> None:
        self.clients.pop(client.id, None)
        self.bus.publish("info", "audio", f"audio stream detached ({self.client_count} client(s))")

    def _broadcast(self, message: dict) -> None:
        """Fan one frame out to every attached client.

        Audio is not telemetry: a dropped log line is a missing log line, but a
        dropped PCM frame is a hole in the middle of a word, and the operator
        hears the assistant stutter with nothing anywhere saying why. So the
        old "make room by discarding the oldest" policy is gone. If a client is
        genuinely this far behind it is broken, and that gets said out loud
        rather than quietly corrupting what it does receive.
        """
        for client in list(self.clients.values()):
            try:
                client.queue.put_nowait(message)
            except asyncio.QueueFull:
                if not client.overflowed:
                    client.overflowed = True
                    self.bus.publish(
                        "error", "audio",
                        f"client {client.id} is more than {self.CLIENT_QUEUE_FRAMES} frames "
                        "behind - audio is being lost to it",
                    )

    def _next_id(self) -> str:
        self._utterance_seq += 1
        return f"utt-{self._utterance_seq:04d}"

    # -- one-shot synthesis ---------------------------------------------------

    async def speak(self, text: str, interrupt: bool = True,
                    on_first_audio: Callable[[], None] | None = None) -> str:
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
        utterance_id = self._next_id()
        self.last_text = text
        task = asyncio.create_task(
            self._run_utterance(utterance_id, text, on_first_audio),
            name=f"tts-{utterance_id}",
        )
        self._current = task
        await asyncio.gather(task, return_exceptions=True)
        return utterance_id

    # -- streamed synthesis ---------------------------------------------------

    def open_stream(self, on_first_audio: Callable[[], None] | None = None) -> SpeechStream:
        """Begin an utterance that will be fed sentence by sentence."""
        utterance_id = self._next_id()
        stream = SpeechStream(self, utterance_id, asyncio.get_running_loop())
        self.stream_audio_started = False
        self._stream = stream
        task = asyncio.create_task(
            self._run_stream(utterance_id, stream, on_first_audio),
            name=f"tts-stream-{utterance_id}",
        )
        self._current = task
        return stream

    def restart_stream_audio(self) -> None:
        """A tool round invalidated what we had begun saying."""
        if self.stream_audio_started:
            self._broadcast({"type": "tts.flush", "reason": "superseded"})
            self.stream_audio_started = False
            self._play_deadline = None
            self._truncate_spoken()

    async def await_stream(self) -> bool:
        task = self._current
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        stream, self._stream = self._stream, None
        spoke = self.stream_audio_started
        if stream is not None and spoke:
            self.last_text = stream.spoken_text
        return spoke

    async def stop(self, reason: str = "interrupted") -> bool:
        """Barge-in: cancel the in-flight utterance and flush client buffers."""
        stream = self._stream
        if stream is not None:
            stream._closed = True  # noqa: SLF001 - same module, deliberate
            stream.stopped = True
        task = self._current
        if task is None or task.done():
            self._stream = None
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._broadcast({"type": "tts.flush", "reason": reason})
        self._play_deadline = None
        self._truncate_spoken()
        self._stream = None
        self.bus.publish("warn", "tts", f"utterance {reason}")
        return True

    # -- synthesis core -------------------------------------------------------

    async def _synthesise(self, utterance_id: str, text: str, seq_start: int,
                          on_first_audio: Callable[[], None] | None) -> tuple[int, int]:
        """Stream one fragment to the clients. Returns (frames, samples)."""
        frames = 0
        samples = 0
        stop = threading.Event()
        producer: asyncio.Future | None = None
        # Unbounded on purpose. The producer is a worker thread that hands over
        # a whole sentence's frames in a burst as soon as the engine finishes
        # it, far faster than the consumer broadcasts them; a bounded queue
        # here therefore overflows on any fragment longer than the bound and,
        # under the old drop-oldest policy, deleted audio from the middle of
        # the sentence. The size is not actually unbounded in practice - one
        # fragment is capped at `speech_max_chars`, a few seconds of PCM.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def safe_put(item) -> None:
            """Hand one frame to the loop. Never raises, never discards."""
            queue.put_nowait(item)

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

        try:
            producer = loop.run_in_executor(None, produce)
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, _Errored):
                    raise item.error
                if not isinstance(item, SpeechChunk) or item.pcm.size == 0:
                    continue
                if frames == 0 and seq_start == 0 and on_first_audio is not None:
                    with contextlib.suppress(Exception):
                        on_first_audio()
                frames += 1
                samples += int(item.pcm.size)
                self._broadcast({
                    "type": "tts.chunk",
                    "utterance_id": utterance_id,
                    "seq": seq_start + frames,
                    "sample_rate": item.sample_rate,
                    "data": base64.b64encode(item.pcm.tobytes()).decode("ascii"),
                })
                # Yield control so socket sends stay prompt.
                await asyncio.sleep(0)
        finally:
            stop.set()  # halt the producer thread promptly on cancel/exit
            if producer is not None:
                await asyncio.gather(producer, return_exceptions=True)
        return frames, samples

    def _extend_playback(self, samples: int, sample_rate: int, text: str = "") -> float:
        """Push the playback deadline out by the audio we just streamed."""
        duration = samples / max(1, sample_rate)
        now = time.monotonic()
        start = max(now, self._play_deadline or now)
        self._play_deadline = start + duration
        if text:
            self._spoken.append((self._play_deadline, text))
        return duration

    # -- echo guard -----------------------------------------------------------

    def recent_speech(self, window_s: float) -> str:
        """Everything still audible, or audible within the last `window_s`.

        A microphone hears the room, and a recogniser reports what it heard
        seconds later, so "recent" has to reach back past the end of the
        utterance rather than only covering what is playing right now.
        """
        cutoff = time.monotonic() - max(0.0, window_s)
        return " ".join(text for ends_at, text in self._spoken if ends_at >= cutoff)

    def _truncate_spoken(self) -> None:
        """A barge-in silenced audio early: nothing plays past this instant."""
        now = time.monotonic()
        self._spoken = deque(
            ((min(ends_at, now), text) for ends_at, text in self._spoken),
            maxlen=self._spoken.maxlen,
        )

    async def _drain_playback(self) -> None:
        remaining = (self._play_deadline or 0) - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        self._play_deadline = None

    async def _run_utterance(self, utterance_id: str, text: str,
                             on_first_audio: Callable[[], None] | None = None) -> None:
        started = time.monotonic()
        sample_rate = self.engine.sample_rate
        self._play_deadline = None
        frames = 0

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
            frames, samples = await self._synthesise(utterance_id, text, 0, on_first_audio)
            duration = self._extend_playback(samples, sample_rate, text)
            self._broadcast({
                "type": "tts.end", "utterance_id": utterance_id,
                "frames": frames, "duration_s": round(duration, 2),
            })
            synth_elapsed = time.monotonic() - started
            self.bus.publish(
                "voice", "tts",
                f"utterance {utterance_id} synthesised - {frames} frames, "
                f"{duration:.1f}s audio in {synth_elapsed:.1f}s "
                f"(rtf {synth_elapsed / max(0.01, duration):.2f})",
            )
            await self._drain_playback()
        except asyncio.CancelledError:
            self._broadcast({
                "type": "tts.end", "utterance_id": utterance_id,
                "frames": frames, "cancelled": True,
            })
            raise
        except Exception as exc:  # noqa: BLE001
            self._broadcast({"type": "tts.error", "utterance_id": utterance_id, "detail": str(exc)})
            self.bus.publish("error", "tts", f"synthesis failed: {type(exc).__name__}: {exc}")

    async def _run_stream(self, utterance_id: str, stream: SpeechStream,
                          on_first_audio: Callable[[], None] | None = None) -> None:
        """Consume sentences as they close and speak them in order."""
        started = time.monotonic()
        sample_rate = self.engine.sample_rate
        self._play_deadline = None
        frames = 0
        samples = 0
        fragments = 0

        try:
            while True:
                fragment = await stream._segments.get()  # noqa: SLF001 - same module
                if fragment is None:
                    break
                if not self.stream_audio_started:
                    # Announce only once we genuinely have something to say, so
                    # a directive answered by a tool round never flashes an
                    # empty utterance at the interface.
                    self._broadcast({
                        "type": "tts.start",
                        "utterance_id": utterance_id,
                        "engine": self.engine.name,
                        "voice": getattr(self.engine, "voice", "default"),
                        "sample_rate": sample_rate,
                        "text": fragment,
                    })
                    self.stream_audio_started = True
                    self.bus.publish(
                        "voice", "tts",
                        f"streaming speech with {self.engine.name} - first fragment in "
                        f"{time.monotonic() - started:.2f}s",
                    )
                got_frames, got_samples = await self._synthesise(
                    utterance_id, fragment, frames, on_first_audio if fragments == 0 else None
                )
                frames += got_frames
                samples += got_samples
                fragments += 1
                self._extend_playback(got_samples, sample_rate, fragment)

            if fragments:
                duration = samples / max(1, sample_rate)
                self._broadcast({
                    "type": "tts.end", "utterance_id": utterance_id,
                    "frames": frames, "duration_s": round(duration, 2),
                })
                synth_elapsed = time.monotonic() - started
                self.bus.publish(
                    "voice", "tts",
                    f"utterance {utterance_id} streamed - {fragments} fragment(s), "
                    f"{duration:.1f}s audio in {synth_elapsed:.1f}s "
                    f"(rtf {synth_elapsed / max(0.01, duration):.2f})",
                )
                await self._drain_playback()
        except asyncio.CancelledError:
            if self.stream_audio_started:
                self._broadcast({
                    "type": "tts.end", "utterance_id": utterance_id,
                    "frames": frames, "cancelled": True,
                })
            raise
        except Exception as exc:  # noqa: BLE001
            self._broadcast({"type": "tts.error", "utterance_id": utterance_id, "detail": str(exc)})
            self.bus.publish("error", "tts", f"streamed synthesis failed: {type(exc).__name__}: {exc}")

    async def shutdown(self) -> None:
        await self.stop(reason="shutdown")
