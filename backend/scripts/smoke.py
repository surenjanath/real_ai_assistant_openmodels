#!/usr/bin/env python3
"""End-to-end smoke test for the J.A.R.V.I.S. backend.

Boots nothing itself - point it at a running server:

    python scripts/smoke.py [--url http://127.0.0.1:8000]

Verifies: health, the settings registry (including the extended-thinking
toggle), real host vitals, control-intent routing, barge-in, and - most
importantly - that the vocal engine actually produces audible speech rather
than silently degrading to the fallback synth.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import urllib.request
import wave

import websockets

FAILURES: list[str] = []


def ok(msg: str) -> None:
    print(f"PASS {msg}")


def bad(msg: str) -> None:
    FAILURES.append(msg)
    print(f"FAIL {msg}")


def warn(msg: str) -> None:
    print(f"WARN {msg}")


def _get(url: str, timeout: float = 8.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post(url: str, body: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------- checks ---


async def check_health(base: str) -> dict:
    data = _get(f"{base}/api/health")
    if not data.get("ok"):
        bad(f"health - {data}")
        return {}
    tts = data["engines"]["tts"]
    agents = data["engines"]["agents"]
    ok(f"health - tts={tts['label']} ({tts['mode']}) agents={agents['name']}/{agents['mode']} model={agents['model']}")
    if tts["mode"] != "kokoro":
        warn("vocal engine is the FALLBACK synth - install pykokoro on python 3.12 for real speech")
    if not agents["model"]:
        warn("no crew model - run `ollama pull <model>`")
    elif not agents["verified"]:
        warn(f"model '{agents['model']}' is not installed on the ollama server")
    return data


async def check_vitals(base: str) -> None:
    data = _get(f"{base}/api/vitals")
    if "cpu" not in data or "cores" not in data:
        bad(f"vitals - unexpected shape {list(data)[:6]}")
        return
    if data.get("source") == "psutil" and data.get("mem", 0) <= 0:
        bad("vitals - psutil reported 0% memory, which cannot be right")
        return
    ok(f"vitals - cpu={data['cpu']}% mem={data.get('mem')}% cores={data['cores']} via {data.get('source')}")


async def check_settings(base: str) -> None:
    """Settings registry: read, live voice/speed switch, think toggle, restore."""
    data = _get(f"{base}/api/settings")
    original = data["settings"]
    if not (original["models"] and original["voices"]):
        bad("settings - catalogues empty")
        return
    ok(f"settings read - model={original['model']} installed={len(original['installed'])} voices={len(original['voices'])}")

    # model_verified must reflect what Ollama actually has, never the catalogue.
    if original["model_verified"] and original["model"] not in original["installed"]:
        bad("settings - model_verified is true for a model that is not installed")
    else:
        ok("settings - model_verified agrees with the installed list")

    new_voice = next(v for v in original["voices"] if v != original["voice"])
    result = _post(f"{base}/api/settings", {"voice": new_voice, "speed": 1.15})
    if not (result["ok"] and "voice" in result["applied"] and "speed" in result["applied"]):
        bad(f"settings write - {result}")
    else:
        ok(f"settings write - {result['applied']}")

    # Extended thinking must be togglable and default to off (it is ~28x slower).
    if original.get("think"):
        warn("extended thinking is ON - answers will be far slower")
    if original.get("think_supported"):
        on = _post(f"{base}/api/settings", {"think": True})
        if on["settings"]["think_active"]:
            ok("settings - extended thinking can be enabled")
        else:
            bad(f"settings - think toggle did not take: {on}")
        _post(f"{base}/api/settings", {"think": original.get("think", False)})
    else:
        ok("settings - current model has no thinking mode (nothing to toggle)")

    _post(f"{base}/api/settings", {"voice": original["voice"], "speed": original["speed"]})
    ok("settings restored")


async def check_logs_and_command(base: str) -> None:
    """Telemetry socket: hello frame, vitals, control intent, answer frames."""
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://")
    async with websockets.connect(f"{ws_url}/ws/logs", max_size=None) as ws:
        hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if hello.get("type") != "hello":
            bad(f"logs - first frame was {hello.get('type')}, expected hello")
            return
        ok(f"logs hello - {hello['name']} v{hello['version']} engines={hello['engines']['tts_label']}")

        _post(f"{base}/api/command", {"text": "what time is it"})

        seen: set[str] = set()
        answer = ""
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            except asyncio.TimeoutError:
                break
            seen.add(frame.get("type", ""))
            if frame.get("type") == "answer":
                answer = frame["text"]
                # status/vitals frames follow the answer - keep reading briefly.
                deadline = min(deadline, time.time() + 6)

        if answer:
            ok(f'control intent routed - "{answer[:60]}"')
        else:
            bad("control intent - no answer frame within 60s")
        for expected in ("log", "status"):
            if expected in seen:
                ok(f"logs - received {expected} frames")
            else:
                bad(f"logs - never received a {expected} frame")
        if "vitals" in seen:
            ok("logs - received live vitals frames")
        else:
            warn("logs - no vitals frame observed in the window")


async def check_audio(base: str, wav_path: str | None) -> None:
    """The one that matters: does the assistant actually make sound?"""
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://")
    async with websockets.connect(f"{ws_url}/ws/audio", max_size=None) as ws:
        loop = asyncio.get_running_loop()
        text = "Systems check. All primary functions are nominal."
        loop.run_in_executor(None, lambda: _post(f"{base}/api/speak", {"text": text}, 180))

        engine = None
        sample_rate = 24000
        chunks: list[bytes] = []
        started = time.time()
        first_audio = None
        utterance: str | None = None
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            except asyncio.TimeoutError:
                break
            kind = frame.get("type")
            uid = frame.get("utterance_id")
            # /ws/audio is a broadcast bus: other clients (browser tabs, n8n)
            # can start and cancel their own utterances at any time. Latch onto
            # the first tts.start whose text is ours and ignore everything else,
            # or a stranger's cancelled utterance ends our capture with 0 frames.
            if kind == "tts.start":
                if utterance is None and frame.get("text", "").startswith("Systems check"):
                    utterance = uid
                    engine = frame["engine"]
                    sample_rate = frame["sample_rate"]
                    started = time.time()
                continue
            if utterance is not None and uid != utterance:
                continue
            if kind == "tts.chunk":
                if first_audio is None:
                    first_audio = time.time() - started
                chunks.append(base64.b64decode(frame["data"]))
            elif kind == "tts.error":
                bad(f"audio - engine error: {frame.get('detail')}")
                return
            elif kind == "tts.end" and utterance is not None:
                if frame.get("cancelled"):
                    warn("audio - our utterance was interrupted by another client")
                break

        if not chunks:
            bad("audio - no PCM chunks received")
            return

        pcm = b"".join(chunks)
        samples = len(pcm) // 2
        duration = samples / sample_rate

        # Silence would still "stream" - assert the waveform has real energy.
        peak = 0
        for i in range(0, min(len(pcm), 400_000), 2):
            v = int.from_bytes(pcm[i : i + 2], "little", signed=True)
            peak = max(peak, abs(v))
        peak_norm = peak / 32768

        if duration < 0.8:
            bad(f"audio - only {duration:.2f}s of audio for a full sentence")
        elif peak_norm < 0.05:
            bad(f"audio - stream is effectively silent (peak {peak_norm:.3f})")
        else:
            ok(
                f"audio - {engine}: {len(chunks)} frames, {duration:.2f}s, "
                f"peak {peak_norm:.2f}, first audio in {first_audio:.2f}s"
            )

        if wav_path:
            with wave.open(wav_path, "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(sample_rate)
                out.writeframes(pcm)
            ok(f"audio written to {wav_path} - play it to confirm the voice by ear")


async def check_stop(base: str) -> None:
    """Barge-in should be accepted whether or not anything is in flight."""
    result = _post(f"{base}/api/stop", {})
    if result.get("ok"):
        ok(f"barge-in endpoint - stopped={result.get('stopped')}")
    else:
        bad(f"barge-in endpoint - {result}")


# ------------------------------------------------------------------ main ---


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--wav", default=None, help="write the spoken sample to this .wav")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print(f"-- smoke testing {base}\n")
    try:
        await check_health(base)
        await check_vitals(base)
        await check_settings(base)
        await check_logs_and_command(base)
        await check_audio(base, args.wav)
        await check_stop(base)
    except Exception as exc:  # noqa: BLE001
        bad(f"unhandled {type(exc).__name__}: {exc}")

    print()
    if FAILURES:
        print(f"== {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"   - {f}")
        return 1
    print("== all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
