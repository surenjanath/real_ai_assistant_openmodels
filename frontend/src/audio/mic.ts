"use client";

/**
 * Acoustic ground truth for the microphone.
 *
 * The speech recogniser is deaf to the difference between the operator and the
 * speakers: on an open mic it transcribes J.A.R.V.I.S.'s own answer just as
 * happily as a directive, and the assistant ends up in conversation with
 * itself. Timing cannot settle it either, because Chrome finalises a phrase a
 * second or more after the sound that produced it.
 *
 * So we open our own capture stream alongside the recogniser's, with the
 * browser's acoustic echo canceller enabled. That canceller is fed whatever
 * the page renders to `AudioContext.destination` - which is exactly the TTS
 * playback - so it subtracts the assistant's voice from this stream. What is
 * left is the room. Energy here while the assistant is talking therefore means
 * a *person* is talking: a genuine barge-in, not an echo.
 *
 * This node is never connected to the destination. Nothing captured is ever
 * played back, sent anywhere, or retained - only its loudness is read.
 */

import { audioEngine } from "./engine";

/** Absolute floor: below this the signal is inaudible whatever the room. */
const SILENCE_RMS = 0.008;
/** How far above the tracked noise floor counts as speech. */
const SPEECH_OVER_FLOOR = 3.2;
/**
 * The same margin, while the assistant is talking.
 *
 * Echo cancellation removes most of our own playback but not all of it, and
 * the residue is loudest in the first moment of an utterance, before the noise
 * floor has had time to adapt to it. Demanding a wider margin there is what
 * keeps leaked echo from registering as a barge-in - the cost is that
 * interrupting the assistant takes a slightly firmer voice than talking to it
 * in silence, which is the right way round.
 */
const SPEECH_OVER_FLOOR_DUCKED = 5.5;
/** Sustained energy required before we call it speech, in ms. */
const ONSET_MS = 90;
/** Silence required before the phrase is considered over, in ms. */
const RELEASE_MS = 420;
/** Sampling period of the level meter, in ms. */
const POLL_MS = 40;

export interface MicMonitorOptions {
  /** Fired once per phrase, the moment sustained human speech is detected. */
  onVoiceStart?: () => void;
  /** Fired when the room has been quiet again for `RELEASE_MS`. */
  onVoiceEnd?: () => void;
}

export class MicMonitor {
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private analyser: AnalyserNode | null = null;
  private buffer = new Float32Array(1024);
  private timer: ReturnType<typeof setInterval> | undefined;
  private options: MicMonitorOptions = {};

  /** Slow-rising, fast-falling estimate of the room's noise floor. */
  private floor = 0.01;
  /** Smoothed RMS of the echo-cancelled input, 0..1. */
  private levelValue = 0;
  private loudSince = 0;
  private quietSince = 0;
  private voicing = false;
  private lastVoiceAtMs = 0;
  private failed = false;
  /** In-flight `start()`, so a rapid toggle cannot open two capture streams. */
  private opening: Promise<boolean> | null = null;
  /** Bumped by `stop()`, so a capture that was granted after we were told to
   *  stop is released rather than left open behind the user's back. */
  private generation = 0;

  /** True once a stream is open and the meter is running. */
  get running(): boolean {
    return this.stream !== null;
  }

  /** True while a person is audibly speaking into the microphone. */
  get voice(): boolean {
    return this.voicing;
  }

  /** Wall-clock ms of the last moment human speech was detected. */
  get lastVoiceAt(): number {
    return this.lastVoiceAtMs;
  }

  /** Smoothed echo-cancelled input level, 0..1. */
  get level(): number {
    return this.levelValue;
  }

  /** True once permission or hardware has definitively refused us. */
  get unavailable(): boolean {
    return this.failed;
  }

  /**
   * Was a person speaking at any point since `sinceMs`?
   *
   * The onset detector needs `ONSET_MS` of energy before it commits, so a
   * phrase that has only just begun still reads as voice here - `lastVoiceAt`
   * is stamped from the first loud sample, not from the commit.
   */
  voiceSince(sinceMs: number): boolean {
    return this.voicing || this.lastVoiceAtMs >= sinceMs;
  }

  start(options: MicMonitorOptions = {}): Promise<boolean> {
    this.options = options;
    if (this.stream) return Promise.resolve(true);
    if (this.failed) return Promise.resolve(false);
    // `open` awaits getUserMedia, and `this.stream` stays null across that
    // await - so without this every toggle during the permission prompt would
    // open another capture stream.
    if (!this.opening) {
      this.opening = this.open().finally(() => {
        this.opening = null;
      });
    }
    return this.opening;
  }

  private async open(): Promise<boolean> {
    const generation = this.generation;
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      this.failed = true;
      return false;
    }
    const ctx = audioEngine.ensure();
    if (!ctx) {
      this.failed = true;
      return false;
    }
    // A suspended context runs no analysis at all, so the meter would sit at
    // zero and report a silent room however loudly it was spoken to.
    audioEngine.unlock();
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // The whole point: let the browser subtract our own playback.
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
    } catch {
      this.failed = true;
      return false;
    }
    if (generation !== this.generation) {
      // Stopped while the permission prompt was up. Hand the microphone back.
      for (const track of stream.getTracks()) track.stop();
      return false;
    }
    this.stream = stream;
    this.source = ctx.createMediaStreamSource(this.stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.2;
    // Analysis only - connecting this to the destination would put the room
    // back through the speakers and build the feedback loop we are here to
    // prevent.
    this.source.connect(analyser);
    this.analyser = analyser;
    this.buffer = new Float32Array(analyser.fftSize);
    this.floor = 0.01;
    this.timer = setInterval(() => this.sample(), POLL_MS);
    return true;
  }

  stop(): void {
    this.generation++;
    clearInterval(this.timer);
    this.timer = undefined;
    try {
      this.source?.disconnect();
    } catch {
      /* already torn down */
    }
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.stream = null;
    this.source = null;
    this.analyser = null;
    this.voicing = false;
    this.levelValue = 0;
  }

  private sample(): void {
    const analyser = this.analyser;
    if (!analyser) return;
    analyser.getFloatTimeDomainData(this.buffer);
    let sum = 0;
    for (let i = 0; i < this.buffer.length; i++) sum += this.buffer[i] * this.buffer[i];
    const rms = Math.sqrt(sum / this.buffer.length);
    this.levelValue += (rms - this.levelValue) * 0.5;

    // Fast down, slow up: a passing noise raises the floor only gradually,
    // so it cannot mask the speech that follows it.
    this.floor = rms < this.floor ? rms : Math.min(this.floor * 1.03 + 0.00025, rms);

    const now = Date.now();
    const margin = audioEngine.audibleWithin(now, 200)
      ? SPEECH_OVER_FLOOR_DUCKED
      : SPEECH_OVER_FLOOR;
    const loud = rms > Math.max(SILENCE_RMS, this.floor * margin);

    if (loud) {
      this.quietSince = 0;
      if (!this.loudSince) this.loudSince = now;
      this.lastVoiceAtMs = now;
      if (!this.voicing && now - this.loudSince >= ONSET_MS) {
        this.voicing = true;
        this.options.onVoiceStart?.();
      }
    } else {
      this.loudSince = 0;
      if (!this.quietSince) this.quietSince = now;
      if (this.voicing && now - this.quietSince >= RELEASE_MS) {
        this.voicing = false;
        this.options.onVoiceEnd?.();
      }
    }
  }
}

/** Client-side singleton - one capture stream is enough for the whole app. */
export const micMonitor = new MicMonitor();
