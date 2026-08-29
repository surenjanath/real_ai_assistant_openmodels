/**
 * Web Audio playback engine (Phase 4).
 *
 * Receives streamed int16 PCM chunks from /ws/audio, schedules them gaplessly
 * on a single AudioContext, and exposes an AnalyserNode whose frequency and
 * time-domain data drive the hologram via `audioLevels`.
 *
 * Autoplay policy: the context may start suspended. We track the real
 * `ctx.state` (rather than assuming `resume()` succeeded) and report it, so
 * the interface can show a "click to enable audio" affordance instead of
 * silently dropping the assistant's voice.
 *
 * Barge-in: `flush()` stops every scheduled source immediately, which is what
 * makes "stop" feel instant rather than "stop after the buffer drains".
 */

import { audioLevels, BAND_COUNT, WAVE_COUNT } from "./levels";

const SMOOTH_UP = 0.5;
const SMOOTH_DOWN = 0.09;
/** Input gain on raw analyser energy - the hologram should feel ALIVE. */
const SENSITIVITY = 1.6;
/**
 * Gain multiplier while the operator is talking over the assistant.
 *
 * Not silence: the answer keeps playing quietly, so if the interruption turns
 * out to have been a cough or a door, nothing was lost. The full stop only
 * happens once a transcript confirms a person actually said something.
 */
const DUCK = 0.22;
/** Seconds for the gain to reach a new level - fast, but not a click. */
const GAIN_RAMP = 0.05;

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private freq: Uint8Array<ArrayBuffer> = new Uint8Array(new ArrayBuffer(128));
  private time: Uint8Array<ArrayBuffer> = new Uint8Array(new ArrayBuffer(1024));
  private nextTime = 0;
  /** Sources still scheduled, so barge-in can stop them. */
  private scheduled = new Set<AudioBufferSourceNode>();
  /** Log-spaced bin ranges for the visualizer bands (computed once). */
  private bandRanges: Array<[number, number]> = [];
  private onStateChange: ((unlocked: boolean) => void) | null = null;
  private volume = 0.9;
  private muted = false;
  private ducked = false;
  /**
   * Accumulated phase of the idle "breathing" animation.
   *
   * Integrated rather than derived from the clock, because its rate changes
   * with `thinking`. `sin(elapsed * rate)` looks continuous only while `rate`
   * is constant: change the rate and the argument jumps by `elapsed * delta`,
   * which after a few minutes of uptime is hundreds of radians. The hologram
   * appeared to restart its animation the instant an answer finished, exactly
   * when `thinking` fell back to zero. Integrating `dt * rate` instead means
   * the rate can move freely and the phase never does.
   */
  private idlePhase = 0;
  /**
   * Wall-clock ms at which the last scheduled audio stops sounding.
   *
   * Deliberately separate from `pending`, which is measured against the audio
   * clock and is reset to zero by `reset()` at the start of every utterance
   * and momentarily between fragments. The echo guard needs a monotonic "the
   * room was loud until at least here" mark that survives both, because a
   * recogniser reports what it heard long after the sound has stopped.
   */
  private outputUntil = 0;

  /** Lazily build the graph - safe to call repeatedly. */
  ensure(): AudioContext | null {
    if (typeof window === "undefined") return null;
    if (this.ctx) return this.ctx;
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    const ctx = new Ctor({ latencyHint: "interactive" });
    const gain = ctx.createGain();
    gain.gain.value = this.volume;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.7;
    gain.connect(analyser);
    analyser.connect(ctx.destination);
    this.ctx = ctx;
    this.gain = gain;
    this.analyser = analyser;
    this.freq = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount));
    this.time = new Uint8Array(new ArrayBuffer(analyser.fftSize));
    this.nextTime = ctx.currentTime;

    // Log-spaced bands across the meaningful spectrum (skip bin 0 = DC).
    // Kokoro speech lives mostly under ~8kHz, so cap the top bin rather than
    // spending half the ring on empty high-frequency bins.
    const usable = Math.floor(analyser.frequencyBinCount * 0.55);
    this.bandRanges = [];
    for (let b = 0; b < BAND_COUNT; b++) {
      const lo = Math.max(1, Math.floor(Math.pow(usable, b / BAND_COUNT)));
      const hi = Math.max(lo + 1, Math.floor(Math.pow(usable, (b + 1) / BAND_COUNT)));
      this.bandRanges.push([lo, Math.min(hi, analyser.frequencyBinCount)]);
    }

    ctx.onstatechange = () => this.onStateChange?.(ctx.state === "running");
    return ctx;
  }

  /** Register a listener for autoplay lock/unlock transitions. */
  watchState(cb: (unlocked: boolean) => void): void {
    this.onStateChange = cb;
    cb(this.ctx?.state === "running");
  }

  /** Call from any user gesture to satisfy autoplay policy. */
  unlock(): void {
    const ctx = this.ensure();
    if (!ctx) return;
    if (ctx.state !== "running") {
      void ctx.resume().then(
        () => this.onStateChange?.(ctx.state === "running"),
        () => this.onStateChange?.(false),
      );
    } else {
      this.onStateChange?.(true);
    }
  }

  get unlocked(): boolean {
    return this.ctx?.state === "running";
  }

  setVolume(value: number): void {
    this.volume = Math.max(0, Math.min(1, value));
    this.applyGain();
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    this.applyGain();
  }

  /**
   * Drop to a murmur because someone in the room started talking.
   *
   * This is the immediate half of barge-in. Waiting for a transcript before
   * reacting means the assistant talks over the operator for the second or so
   * the recogniser takes to make up its mind, which is precisely the moment
   * the interruption needed to be acknowledged.
   */
  setDucked(ducked: boolean): void {
    if (this.ducked === ducked) return;
    this.ducked = ducked;
    this.applyGain();
  }

  private applyGain(): void {
    if (!this.gain) return;
    const target = this.muted ? 0 : this.volume * (this.ducked ? DUCK : 1);
    const ctx = this.ctx;
    if (!ctx) {
      this.gain.gain.value = target;
      return;
    }
    // Ramped, not assigned: a step change in gain mid-waveform is an audible
    // click, and this fires every time the operator opens their mouth.
    this.gain.gain.cancelScheduledValues(ctx.currentTime);
    this.gain.gain.setTargetAtTime(target, ctx.currentTime, GAIN_RAMP);
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
    for (let i = 0; i < frameCount; i++) channel[i] = int16[i] / 32768;

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.gain);
    // Drift guard: never schedule into the past, keep a 60ms safety margin.
    const now = ctx.currentTime;
    if (this.nextTime < now + 0.06) this.nextTime = now + 0.06;
    source.start(this.nextTime);
    this.nextTime += frameCount / sampleRate;
    // Only while the context is actually running. A suspended context still
    // advances the schedule but emits no sound, and claiming the room was
    // loud when it was silent would have the echo guard discard everything
    // the operator says while audio is autoplay-locked.
    if (ctx.state === "running") {
      this.outputUntil = Math.max(
        this.outputUntil,
        Date.now() + Math.max(0, this.nextTime - ctx.currentTime) * 1000,
      );
    }
    this.scheduled.add(source);
    source.onended = () => this.scheduled.delete(source);
  }

  /** Wall-clock ms at which the speakers fall silent (past = silent now). */
  get speakingUntil(): number {
    return this.outputUntil;
  }

  /**
   * Was the assistant audible at any point in the window ending now?
   *
   * `hangoverMs` extends the window past the last sample so that room
   * reverberation, and the recogniser's own buffering, are still covered.
   */
  audibleWithin(sinceMs: number, hangoverMs = 0): boolean {
    return this.outputUntil + hangoverMs >= sinceMs;
  }

  /** Barge-in: stop everything already scheduled, immediately. */
  flush(): void {
    for (const source of this.scheduled) {
      try {
        source.onended = null;
        source.stop();
      } catch {
        /* already stopped */
      }
    }
    this.scheduled.clear();
    this.reset();
    // Barge-in cut the sound off here, so the room is quiet from now - not
    // from wherever the schedule happened to reach.
    this.outputUntil = Math.min(this.outputUntil, Date.now());
    // Nothing is playing, so there is nothing to duck; leaving it engaged
    // would start the next utterance at a whisper.
    this.setDucked(false);
  }

  /** Drop the scheduling cursor back to now (does not stop live sources). */
  reset(): void {
    if (this.ctx) this.nextTime = this.ctx.currentTime;
  }

  /** Seconds of audio still queued ahead of the playhead. */
  get pending(): number {
    if (!this.ctx) return 0;
    return Math.max(0, this.nextTime - this.ctx.currentTime);
  }

  /** Called every animation frame from the 3D loop. */
  tick(dt: number, elapsed: number): void {
    audioLevels.kick *= Math.exp(-dt * 3.2);

    // Ease `thinking` toward its target rather than letting the transport step
    // it. It scales several idle amplitudes, and a hard 1 -> 0 the moment an
    // answer lands reads as the visual restarting.
    audioLevels.thinking +=
      (audioLevels.thinkingTarget - audioLevels.thinking) * Math.min(1, dt * 3.5);

    // Advanced every frame, including while speaking, so the idle animation
    // resumes from where it was rather than from wherever the wall clock is.
    this.idlePhase = (this.idlePhase + dt * (1.4 + audioLevels.thinking * 3.5)) % (Math.PI * 2);

    const analyser = this.analyser;
    const speaking = this.pending > 0.01;

    let level = 0;
    let bass = 0;
    let mid = 0;
    let treble = 0;

    if (analyser && speaking) {
      analyser.getByteFrequencyData(this.freq);
      analyser.getByteTimeDomainData(this.time);

      const n = this.freq.length;
      const bEnd = Math.floor(n * 0.06);
      const mEnd = Math.floor(n * 0.3);
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
      treble = Math.min(1, (t / Math.max(1, n - mEnd)) * SENSITIVITY * 1.5);
      level = Math.min(1, bass * 0.5 + mid * 0.4 + treble * 0.3);

      const bands = audioLevels.bands;
      for (let band = 0; band < BAND_COUNT; band++) {
        const [lo, hi] = this.bandRanges[band] ?? [1, 2];
        let sum = 0;
        for (let i = lo; i < hi; i++) sum += this.freq[i] / 255;
        // Higher bands carry less energy - tilt them up so the ring reads evenly.
        const tilt = 1 + (band / BAND_COUNT) * 1.1;
        const value = Math.min(1, (sum / (hi - lo)) * tilt * SENSITIVITY);
        bands[band] += (value - bands[band]) * Math.min(1, dt * 18);
      }

      // Downsample the time-domain data into the waveform ring.
      const wave = audioLevels.wave;
      const step = this.time.length / WAVE_COUNT;
      for (let i = 0; i < WAVE_COUNT; i++) {
        const v = (this.time[Math.floor(i * step)] - 128) / 128;
        wave[i] += (v - wave[i]) * Math.min(1, dt * 30);
      }
    } else {
      // Idle: gentle breathing so the core feels alive between utterances.
      const think = audioLevels.thinking;
      const pulse = 0.5 + 0.5 * Math.sin(this.idlePhase);
      level = 0.05 + 0.04 * pulse + think * 0.14;
      bass = 0.2 + 0.12 * (0.5 + 0.5 * Math.sin(elapsed * 0.9)) + think * 0.2;
      treble = 0.1 + think * 0.1;
      mid = level;

      const bands = audioLevels.bands;
      for (let band = 0; band < BAND_COUNT; band++) {
        const idle =
          0.05 +
          0.045 * (0.5 + 0.5 * Math.sin(elapsed * 1.7 + band * 0.35)) +
          0.03 * (0.5 + 0.5 * Math.sin(elapsed * 0.7 - band * 0.19)) +
          think * 0.18 * (0.5 + 0.5 * Math.sin(elapsed * 5 - band * 0.5));
        bands[band] += (idle - bands[band]) * Math.min(1, dt * 6);
      }
      const wave = audioLevels.wave;
      for (let i = 0; i < WAVE_COUNT; i++) {
        const idle =
          Math.sin(elapsed * 1.6 + i * 0.13) * (0.05 + think * 0.16) +
          Math.sin(elapsed * 3.7 - i * 0.05) * 0.02;
        wave[i] += (idle - wave[i]) * Math.min(1, dt * 8);
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
