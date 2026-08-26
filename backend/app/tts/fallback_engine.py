"""Dependency-free fallback vocal engine.

When Kokoro weights are unavailable (fresh clone, CI, a thin container) this
engine synthesises a robotic formant "console voice" with numpy only. It speaks
the same streaming protocol - int16 PCM chunks - so the entire audio-reactive
pipeline (WebSocket streaming -> Web Audio API -> particle sphere) is exercised
end to end. On a properly provisioned host the Kokoro engine takes over
automatically.

Phonetics: words are split into pseudo-syllables; each syllable is a sum of
three formant sines on a decaying glottal pitch contour, with noise bursts for
fricatives and pauses at punctuation. Deterministic per text (hash seeded).
"""

from __future__ import annotations

import hashlib
from typing import Iterator

import numpy as np

from ..config import settings
from .base import SpeechChunk, to_frames

_VOWELS = set("aeiouy")
_FRICATIVES = set("sfhxz")


class FallbackEngine:
    name = "fallback-synth"
    sample_rate = settings.sample_rate

    def __init__(self, voice: str | None = None) -> None:
        self.voice = voice or settings.tts_voice
        # Male voices (am_/bm_) get a lower fundamental than female (af_/bf_).
        self._f0 = 128.0 if self.voice.startswith(("am_", "bm_")) else 196.0

    # -- public API ---------------------------------------------------------

    def stream(self, text: str) -> Iterator[SpeechChunk]:
        clean = " ".join(text.split())[: settings.tts_max_chars]
        words = self._words(clean)
        if not words:
            words = ["standby"]
        pcm = self._render(words)
        frames = to_frames(pcm)
        for i, frame in enumerate(frames):
            yield SpeechChunk(pcm=frame, sample_rate=self.sample_rate, last=i == len(frames) - 1)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _words(text: str) -> list[str]:
        return [w for w in text.replace("\n", " ").split(" ") if w.strip()]

    def _render(self, words: list[str]) -> np.ndarray:
        sr = self.sample_rate
        parts: list[np.ndarray] = []
        total_words = max(1, len(words))

        for wi, word in enumerate(words):
            seed = int(hashlib.blake2b(word.encode("utf-8"), digest_size=4).hexdigest(), 16)
            rng = np.random.default_rng(seed)
            syllables = self._syllable_count(word)
            # Gentle declination across the utterance, like natural speech.
            progress = wi / total_words
            f0 = self._f0 * (1.06 - 0.12 * progress)

            for si in range(syllables):
                dur = float(rng.uniform(0.085, 0.165))
                n = int(sr * dur)
                t = np.arange(n) / sr
                syl_prog = si / max(1, syllables)
                pitch = f0 * (1.0 - 0.05 * syl_prog) * (1.0 - 0.18 * (t / max(t[-1], 1e-6)))
                # Vary formants per syllable for pseudo-articulation.
                shift = 1.0 + 0.22 * float(rng.uniform(-1, 1))
                f1, f2, f3 = 620 * shift, 1240 / shift, 2550 * shift
                sig = (
                    0.52 * np.sin(2 * np.pi * pitch * t)
                    + 0.26 * np.sin(2 * np.pi * pitch * 2.0 * t + 0.6)
                    + 0.30 * np.sin(2 * np.pi * f1 * t)
                    + 0.18 * np.sin(2 * np.pi * f2 * t + 1.2)
                    + 0.08 * np.sin(2 * np.pi * f3 * t + 2.1)
                )
                # Slight vibrato / jitter so it feels alive.
                sig *= 1.0 + 0.06 * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 6))
                env = np.minimum(1.0, np.sin(np.pi * np.minimum(t / dur, 1.0)) ** 0.6)
                parts.append(sig * env * 0.30)

            # Fricative burst for s/sh/f/x words.
            letters = [c for c in word.lower() if c.isalpha()]
            if letters and letters[-1] in _FRICATIVES:
                n = int(sr * 0.07)
                noise = np.random.default_rng(seed + 7).standard_normal(n)
                kernel = np.array([1.0, 2.0, 1.0]) / 4.0
                noise = np.convolve(noise, kernel, mode="same")
                env = np.hanning(n) ** 0.5
                parts.append(noise * env * 0.10)

            # Inter-word pause; longer at punctuation.
            pause = 0.035
            if word.endswith((",", ";", ":")):
                pause = 0.14
            elif word.endswith((".", "!", "?", ";")):
                pause = 0.22
            parts.append(np.zeros(int(sr * pause), dtype=np.float32))

        audio = np.concatenate(parts) if parts else np.zeros(sr, dtype=np.float32)
        # Final fade-out to avoid clicks.
        fade = int(sr * 0.02)
        if audio.size > 2 * fade:
            audio[-fade:] *= np.linspace(1.0, 0.0, fade)
        max_abs = float(np.max(np.abs(audio))) or 1.0
        audio = audio * (0.88 / max_abs)
        return (audio * 32767.0).astype(np.int16)

    @staticmethod
    def _syllable_count(word: str) -> int:
        letters = [c for c in word.lower() if c.isalpha()]
        if not letters:
            return 1
        count = 0
        prev_vowel = False
        for c in letters:
            is_vowel = c in _VOWELS
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(1, min(count, 5))
