/**
 * Web Audio playback engine (Phase 4).
 *
 * Receives streamed int16 PCM chunks from /ws/audio, schedules them gaplessly
 * on a single AudioContext, and exposes an AnalyserNode whose frequency data
 * drives the particle sphere via `audioLevels` (bands, bass/mid/treble, kick).
 *
 * Handles the browser autoplay policy: if the context starts suspended (no
 * user gesture yet) buffers are still scheduled against the frozen clock and
 * will play the moment `resume()` succeeds (first click / keypress).
 */

import { audioLevels, BAND_COUNT } from "./levels";

const SMOOTH_UP = 0.4;
const SMOOTH_DOWN = 0.08;
/** Input gain on raw analyser energy - the hologram should feel ALIVE. */
const SENSITIVITY = 1.55;

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private freq: Uint8Array<ArrayBuffer> = new Uint8Array(128);
  private nextTime = 0;
  private unlocked = false;
  /** Log-spaced bin ranges for the 24 visualizer bands (computed once). */
  private bandRanges: Array<[number, number]> = [];

  /** Lazily build the graph - safe to call repeatedly. */
  ensure(): AudioContext | null {
    if (typeof window === "undefined") return null;
    if (this.ctx) return this.ctx;
    const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    const ctx = new Ctor({ latencyHint: "interactive" });
    const gain = ctx.createGain();
    gain.gain.value = 0.9;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.72;
    gain.connect(analyser);
    analyser.connect(ctx.destination);
    this.ctx = ctx;
    this.gain = gain;
    this.analyser = analyser;
    this.freq = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount));
    this.nextTime = ctx.currentTime;
    // Log-spaced bands across the meaningful spectrum (skip bin 0 = DC).
    const bins = analyser.frequencyBinCount;
    this.bandRanges = [];
    for (let b = 0; b < BAND_COUNT; b++) {
      const lo = Math.max(1, Math.floor(Math.pow(bins - 1, b / BAND_COUNT)));
      const hi = Math.max(lo + 1, Math.floor(Math.pow(bins - 1, (b + 1) / BAND_COUNT)));
      this.bandRanges.push([lo, Math.min(hi, bins)]);
    }
    return ctx;
  }

  /** Call from any user gesture to satisfy autoplay policy. */
  unlock(): void {
    const ctx = this.ensure();
    if (!ctx) return;
    if (ctx.state === "suspended") void ctx.resume();
    this.unlocked = true;
  }

  /** Transient impulse - call when an utterance starts. */
  kick(): void {
    audioLevels.kick = 1;
  }

  /** Feed one PCM chunk (int16 little-endian mono) into the schedule. */
  push(int16: Int16Array, sampleRate: number): void {
    const ctx = this.ensure();
    if (!ctx || !this.gain || int16.length === 0) return;
    if (ctx.state === "suspended") void ctx.resume();

    const frameCount = int16.length;
    const buffer = ctx.createBuffer(1, frameCount, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < frameCount; i++) {
      channel[i] = int16[i] / 32768;
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.gain);
    // Drift guard: never schedule into the past, keep a 60ms safety margin.
    const now = ctx.currentTime;
    if (this.nextTime < now + 0.06) this.nextTime = now + 0.06;
    source.start(this.nextTime);
    this.nextTime += frameCount / sampleRate;
  }

  /** Drop anything not yet played (e.g. utterance cancelled). */
  reset(): void {
    if (this.ctx) this.nextTime = this.ctx.currentTime;
  }

  /** Called every animation frame from the 3D loop. */
  tick(dt: number, elapsed: number): void {
    // Kick decays fast but visibly.
    audioLevels.kick *= Math.exp(-dt * 3.2);

    const analyser = this.analyser;
    const speaking = !!this.ctx && this.nextTime > this.ctx.currentTime + 0.01;

    let level = 0;
    let bass = 0;
    let mid = 0;
    let treble = 0;

    if (analyser && speaking) {
      analyser.getByteFrequencyData(this.freq);
      const n = this.freq.length;
      const bEnd = Math.floor(n * 0.08);
      const mEnd = Math.floor(n * 0.4);
      let b = 0;
      let m = 0;
      let t = 0;
      for (let i = 0; i < n; i++) {
        const v = this.freq[i] / 255;
        if (i < bEnd) b += v;
        else if (i < mEnd) m += v;
        else t += v;
      }
      bass = Math.min(1, (b / Math.max(1, bEnd)) * SENSITIVITY * 1.15);
      mid = Math.min(1, (m / Math.max(1, mEnd - bEnd)) * SENSITIVITY);
      treble = Math.min(1, (t / Math.max(1, n - mEnd)) * SENSITIVITY * 1.3);
      level = Math.min(1, bass * 0.55 + mid * 0.35 + treble * 0.25);

      // Refresh the visualizer bands.
      const bands = audioLevels.bands;
      for (let band = 0; band < BAND_COUNT; band++) {
        const [lo, hi] = this.bandRanges[band] ?? [1, 2];
        let sum = 0;
        for (let i = lo; i < hi; i++) sum += this.freq[i] / 255;
        // Higher bands carry less energy - tilt them up.
        const tilt = 1 + (band / BAND_COUNT) * 0.9;
        const value = Math.min(1, (sum / (hi - lo)) * tilt * SENSITIVITY);
        bands[band] += (value - bands[band]) * Math.min(1, dt * 14);
      }
    } else {
      // Idle: gentle breathing so the core feels alive between utterances.
      level = 0.05 + 0.035 * (0.5 + 0.5 * Math.sin(elapsed * 1.4));
      bass = 0.25 + 0.15 * (0.5 + 0.5 * Math.sin(elapsed * 0.9));
      treble = 0.12;
      mid = level;
      const bands = audioLevels.bands;
      for (let band = 0; band < BAND_COUNT; band++) {
        const idle =
          0.06 +
          0.05 * (0.5 + 0.5 * Math.sin(elapsed * 1.8 + band * 0.55)) +
          0.03 * (0.5 + 0.5 * Math.sin(elapsed * 0.7 - band * 0.3));
        bands[band] += (idle - bands[band]) * Math.min(1, dt * 6);
      }
    }

    const kUp = 1 - Math.pow(1 - SMOOTH_UP, dt * 60);
    const kDown = 1 - Math.pow(1 - SMOOTH_DOWN, dt * 60);
    const lerpTo = (current: number, target: number) =>
      current + (target - current) * (target > current ? kUp : kDown);

    audioLevels.level = lerpTo(audioLevels.level, level);
    audioLevels.bass = lerpTo(audioLevels.bass, bass);
    audioLevels.mid = lerpTo(audioLevels.mid, mid);
    audioLevels.treble = lerpTo(audioLevels.treble, treble);
    audioLevels.speaking = speaking;
  }

  get isUnlocked(): boolean {
    return this.unlocked;
  }
}

/** Client-side singleton. */
export const audioEngine = new AudioEngine();

/** Decode a base64 int16 LE PCM payload. */
export function decodePcm16LE(base64: string): Int16Array {
  const raw = atob(base64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  const aligned = bytes.length % 2 === 0 ? bytes : bytes.subarray(0, bytes.length - 1);
  return new Int16Array(aligned.buffer, aligned.byteOffset, aligned.length / 2);
}
