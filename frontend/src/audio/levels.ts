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
  /** true while scheduled audio is still playing */
  speaking: boolean;
}

export const audioLevels: AudioLevels = {
  level: 0,
  bass: 0,
  mid: 0,
  treble: 0,
  speaking: false,
};
