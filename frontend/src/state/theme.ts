/**
 * Scene palette, shared between the CSS colour schemes and the WebGL scene.
 *
 * A theme that retinted the panels but left a giant blue hologram in the
 * middle of them would look broken, so the accent family lives here as
 * mutable `THREE.Color` instances that the scene lerps toward every frame.
 * Switching themes therefore *cross-fades* the reactor rather than snapping
 * it, and — as everywhere else in this interface — costs no React renders.
 */

import * as THREE from "three";

export interface ScenePalette {
  /** hottest highlight: particle peaks, the seed crystal */
  core: string;
  /** mid accent: spectrum bars, oscilloscope */
  hot: string;
  /** rim/shadow tone of the particle field */
  rim: string;
  /** instrument rings and ticks */
  ring: string;
  /** far starfield dust */
  dust: string;
}

export const SCENE_THEMES: Record<string, ScenePalette> = {
  arc: {
    core: "#eaf6ff",
    hot: "#8fd6ff",
    rim: "#3f7fd8",
    ring: "#7fb8ff",
    dust: "#7fa8dd",
  },
  crimson: {
    core: "#fff0f2",
    hot: "#ff9fae",
    rim: "#c8384f",
    ring: "#ff8fa0",
    dust: "#d98d9c",
  },
  emerald: {
    core: "#eefff7",
    hot: "#7ff0c0",
    rim: "#1f9e6a",
    ring: "#5fe0ac",
    dust: "#7fc9a9",
  },
  amber: {
    core: "#fff6e8",
    hot: "#ffd08a",
    rim: "#c9832a",
    ring: "#ffc46a",
    dust: "#d9b382",
  },
  violet: {
    core: "#f6f1ff",
    hot: "#c4aeff",
    rim: "#6d4bd8",
    ring: "#b198ff",
    dust: "#a894d8",
  },
  // Not literally grey: a hologram with zero saturation reads as dead pixels
  // rather than as light, so the core keeps the faintest cool cast.
  mono: {
    core: "#ffffff",
    hot: "#dfe5ec",
    rim: "#6d7683",
    ring: "#c2cad6",
    dust: "#9aa3af",
  },
};

/** Live values the scene reads. Mutated in place — never reassigned. */
export const sceneColors = {
  core: new THREE.Color(SCENE_THEMES.arc.core),
  hot: new THREE.Color(SCENE_THEMES.arc.hot),
  rim: new THREE.Color(SCENE_THEMES.arc.rim),
  ring: new THREE.Color(SCENE_THEMES.arc.ring),
  dust: new THREE.Color(SCENE_THEMES.arc.dust),
};

/** Where the scene is heading; `sceneColors` eases toward this each frame. */
const target = {
  core: new THREE.Color(SCENE_THEMES.arc.core),
  hot: new THREE.Color(SCENE_THEMES.arc.hot),
  rim: new THREE.Color(SCENE_THEMES.arc.rim),
  ring: new THREE.Color(SCENE_THEMES.arc.ring),
  dust: new THREE.Color(SCENE_THEMES.arc.dust),
};

export function applySceneTheme(name: string): void {
  const palette = SCENE_THEMES[name] ?? SCENE_THEMES.arc;
  target.core.set(palette.core);
  target.hot.set(palette.hot);
  target.rim.set(palette.rim);
  target.ring.set(palette.ring);
  target.dust.set(palette.dust);
}

/** Ease the live palette toward the target. Called once per rendered frame. */
export function stepSceneTheme(dt: number): void {
  const k = Math.min(1, dt * 3.2);
  sceneColors.core.lerp(target.core, k);
  sceneColors.hot.lerp(target.hot, k);
  sceneColors.rim.lerp(target.rim, k);
  sceneColors.ring.lerp(target.ring, k);
  sceneColors.dust.lerp(target.dust, k);
}
