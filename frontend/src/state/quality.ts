/**
 * Render quality tiers for the holographic scene.
 *
 * The interface was built on a machine that can push 24,000 GPU-displaced
 * particles at 60 fps. Plenty cannot, and on those the whole thing degrades
 * into a slideshow — which also starves the audio callback that drives the
 * displacement, so the hologram stops reacting to the voice as well. That is a
 * much worse failure than simply drawing fewer points.
 *
 * So the tier is chosen from what the device advertises, then corrected by
 * what it actually delivers (see `PerfGuard` in Scene.tsx). Like `audioLevels`,
 * this is a mutable singleton rather than React state: the WebGL tree must
 * never re-render because a frame was slow.
 */

export const TIERS = ["low", "balanced", "high"] as const;
export type Quality = (typeof TIERS)[number];

export interface QualityProfile {
  /** points in the displaced core sphere */
  particles: number;
  /** background starfield points */
  stars: number;
  /** device-pixel-ratio clamp handed to the renderer */
  dpr: [number, number];
  antialias: boolean;
  /** orbiting satellite count */
  satellites: number;
  label: string;
}

export const PROFILES: Record<Quality, QualityProfile> = {
  low: { particles: 5000, stars: 160, dpr: [1, 1], antialias: false, satellites: 2, label: "low" },
  balanced: { particles: 12000, stars: 340, dpr: [1, 1.5], antialias: true, satellites: 3, label: "balanced" },
  high: { particles: 24000, stars: 520, dpr: [1, 2], antialias: true, satellites: 3, label: "high" },
};

const STORAGE_KEY = "jarvis.quality";

interface NavigatorWithHints extends Navigator {
  deviceMemory?: number;
}

/** A first guess from what the browser will tell us, before any frame is drawn. */
function guess(): Quality {
  if (typeof window === "undefined") return "high";
  const nav = window.navigator as NavigatorWithHints;
  const cores = nav.hardwareConcurrency ?? 4;
  const memory = nav.deviceMemory ?? 8;
  const coarse = window.matchMedia?.("(pointer: coarse)").matches ?? false;
  // A phone or tablet has neither the fill rate nor the thermal headroom, and
  // the scene is barely legible at that size anyway.
  if (coarse || cores <= 4 || memory <= 4) return "low";
  if (cores <= 8 || memory <= 8) return "balanced";
  return "high";
}

type Listener = (tier: Quality, auto: boolean) => void;
const listeners = new Set<Listener>();

/** Mutable singleton - read directly from render code, never subscribed to. */
export const quality = {
  tier: "high" as Quality,
  /** false once the operator has picked a tier by hand */
  auto: true,
  /** measured frames per second, refreshed about once a second */
  fps: 0,
  /** whether the automatic guard has already stepped down this session */
  demoted: false,
};

/** Restore the operator's choice, or fall back to the hardware guess. */
export function initQuality(): Quality {
  let saved: string | null = null;
  try {
    saved = window.localStorage.getItem(STORAGE_KEY);
  } catch {
    // Storage blocked; the guess below is perfectly serviceable.
  }
  if (saved && (TIERS as readonly string[]).includes(saved)) {
    quality.tier = saved as Quality;
    quality.auto = false;
  } else {
    quality.tier = guess();
    quality.auto = true;
  }
  return quality.tier;
}

export function setQuality(tier: Quality, auto = false): void {
  if (quality.tier === tier && quality.auto === auto) return;
  quality.tier = tier;
  quality.auto = auto;
  if (!auto) {
    try {
      window.localStorage.setItem(STORAGE_KEY, tier);
    } catch {
      // As above - the tier simply will not survive a reload.
    }
  }
  listeners.forEach((fn) => fn(tier, auto));
}

/** Hand control back to the hardware guess. */
export function resetQuality(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
  quality.demoted = false;
  setQuality(guess(), true);
}

export function onQualityChange(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** One step down the ladder, or null if we are already at the bottom. */
export function lowerTier(tier: Quality): Quality | null {
  const index = TIERS.indexOf(tier);
  return index > 0 ? TIERS[index - 1] : null;
}

export function profile(): QualityProfile {
  return PROFILES[quality.tier];
}
