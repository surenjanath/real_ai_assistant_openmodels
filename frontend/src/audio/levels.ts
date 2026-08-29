/**
 * Live audio-band levels, shared between the Web Audio analyser and the
 * 3D render loop as a plain mutable object.
 *
 * Deliberately NOT React/Zustand state: mutating it must never trigger a
 * re-render of the Canvas tree (PRD §3 - state management without
 * unnecessary global re-renders of the 3D scene).
 */

export interface AudioLevels {
  /** overall loudness 0..1 (smoothed) */
  level: number;
  /** low-frequency energy 0..1 - drives rotation acceleration + core pulse */
  bass: number;
  /** mid-frequency energy 0..1 - drives particle scale */
  mid: number;
  /** high-frequency energy 0..1 - drives glow / point brightness */
  treble: number;
  /** 64 log-spaced spectrum bands 0..1 - drives the ring visualizer */
  bands: Float32Array;
  /** 128-point time-domain waveform, -1..1 - drives the oscilloscope ring */
  wave: Float32Array;
  /** transient kick 0..1, spikes at utterance start, decays exponentially */
  kick: number;
  /** true while scheduled audio is still playing */
  speaking: boolean;
  /** true while the mic is capturing */
  listening: boolean;
  /** 0..1 ramp toward the "thinking" visual state, eased by the engine */
  thinking: number;
  /** where `thinking` is heading: set by the transport, never read by the scene */
  thinkingTarget: number;
}

export const BAND_COUNT = 64;
export const WAVE_COUNT = 128;

export const audioLevels: AudioLevels = {
  level: 0,
  bass: 0,
  mid: 0,
  treble: 0,
  bands: new Float32Array(BAND_COUNT),
  wave: new Float32Array(WAVE_COUNT),
  kick: 0,
  speaking: false,
  listening: false,
  thinking: 0,
  thinkingTarget: 0,
};
