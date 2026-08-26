#!/usr/bin/env python3
"""End-to-end smoke test for the J.A.R.V.I.S. backend.

Boots nothing itself - point it at a running server:

    python scripts/smoke.py [--url http://127.0.0.1:8000]

Verifies: health endpoint, telemetry WebSocket (logs stream + status frames),
command routing, and TTS audio chunk streaming over /ws/audio.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import urllib.request
import wave

import websockets


async def check_health(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/api/health", timeout=5) as resp:
        data = json.loads(resp.read().decode())
    assert data["ok"], data
    print(f"PASS health - tts={data['engines']['tts']['name']} agents={data['engines']['agents']['name']}")
    return data


async def check_settings(base: str) -> None:
    """Settings registry: read, live voice/speed switch, restore."""
    with urllib.request.urlopen(f"{base}/api/settings", timeout=5) as resp:
        data = json.loads(resp.read().decode())
    original = data["settings"]
    assert original["models"] and original["voices"], "catalogues empty"
    print(f"PASS settings read - model={original['model']} voices={len(original['voices'])}")

    new_voice = next(v for v in original["voices"] if v != original["voice"])
    payload = json.dumps({"voice": new_voice, "speed": 1.15}).encode()
    req = urllib.request.Request(f"{base}/api/settings", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        result = json.loads(resp.read().decode())
    assert result["ok"] and "voice" in result["applied"] and "speed" in result["applied"], result
    print(f"PASS settings write - {result['applied']}")

    restore = json.dumps({"voice": original["voice"], "speed": original["speed"]}).encode()
    req = urllib.request.Request(f"{base}/api/settings", data=restore,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        json.loads(resp.read().decode())
    print("PASS settings restore")


async def check_logs(base: str) -> None:
    async with websockets.connect(base.replace("http", "ws") + "/ws/logs") as ws:
        hello = json.loads(await asyncio.wait_for(ws.recv(), 5))
        assert hello["type"] == "hello", hello
        print(f"PASS logs hello - engines: {hello['engines']}")

        got_log = False
        deadline = time.time() + 12
        while not got_log and time.time() < deadline:
            await ws.send(json.dumps({"type": "command", "text": "run a full system diagnostic"}))
            frame = json.loads(await asyncio.wait_for(ws.recv(), 6))
            if frame["type"] == "log":
                got_log = True
        assert got_log, "no log frames received"
        print(f"PASS logs stream - sample: [{frame['level']}] {frame['source']}: {frame['msg'][:60]}")


async def check_audio(base: str, out_wav: str) -> None:
    async with websockets.connect(base.replace("http", "ws") + "/ws/audio") as ws:
        # A previous utterance may still be streaming (e.g. a crew answer);
        # our speak interrupts it. Track OUR utterance id and ignore others.
        await ws.send(json.dumps({"type": "speak", "text": "All systems nominal. Standing by."}))
        mine: str | None = None
        frames: list[bytes] = []
        sample_rate = 24000
        deadline = time.time() + 30
        finished = False
        while time.time() < deadline:
            frame = json.loads(await asyncio.wait_for(ws.recv(), 10))
            ftype = frame["type"]
            if ftype == "tts.start" and mine is None:
                mine = frame["utterance_id"]
                sample_rate = frame["sample_rate"]
                print(f"PASS audio start - engine={frame['engine']} rate={sample_rate}")
            elif ftype == "tts.chunk" and frame["utterance_id"] == (mine or ""):
                frames.append(base64.b64decode(frame["data"]))
            elif ftype == "tts.end" and frame["utterance_id"] == mine:
                if frame.get("cancelled"):
                    raise AssertionError("our utterance was cancelled by another request")
                finished = True
                break
        assert finished and frames, f"audio stream incomplete ({len(frames)} frames)"
        pcm = b"".join(frames)
        with wave.open(out_wav, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        duration = len(pcm) / 2 / sample_rate
        print(f"PASS audio stream - {len(frames)} frames, {duration:.1f}s -> {out_wav}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--wav", default="/tmp/jarvis_smoke.wav")
    args = parser.parse_args()

    await check_health(args.url)
    await check_settings(args.url)
    await check_logs(args.url)
    await check_audio(args.url, args.wav)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
