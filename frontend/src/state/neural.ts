/**
 * The cognitive graph, mirrored client-side.
 *
 * Like `audioLevels`, this is a plain mutable singleton rather than React or
 * Zustand state: the backend pushes coalesced activation at 20 Hz, and running
 * that through a store would re-render the WebGL tree twenty times a second
 * for no reason. The 3D mesh and the 2D cortex panel both read straight from
 * here inside their own animation frames.
 *
 * The only thing that *is* reactive is the graph version counter, which the
 * components watch to know when the topology changed (a tool node grew) and
 * geometry must be rebuilt.
 */

import type { NeuralEdgeFrame, NeuralNodeFrame } from "@/lib/protocol";

/**
 * The cortex is laid out as a **layered barrel** around the reactor: one
 * horizontal ring per layer, stacked top (sensory) to bottom (motor), bulging
 * outward at the equator.
 *
 * The first attempt scattered nodes over a sphere with a golden-angle offset
 * per layer. It looked like noise: because consecutive layers were rotated
 * against each other, almost every edge became a long arc across the whole
 * shell, and the picture read as random scratches rather than as a network.
 * Aligning the layers turns the same graph into short, mostly-vertical
 * synapses, so the eye can actually follow signal descending through it — and
 * it composes with the existing concentric ring system instead of fighting it.
 */
export const SHELL_RADIUS = 3.35;
/** Half-height of the barrel; keeps the stack inside the camera frustum. */
export const SHELL_HEIGHT = 2.0;
/**
 * How much narrower the end rings are than the equator: 0 is a cylinder, 1
 * tapers the top and bottom rings to a point. SHELL_RADIUS is the equator, so
 * this only ever pulls rings *inward* — the barrel never exceeds its radius.
 */
export const SHELL_BULGE = 0.24;
/** Cap on simultaneously travelling action potentials. */
export const MAX_PULSES = 160;
/** Edge activation decay, per second. */
export const FLOW_DECAY = 1.9;

export interface MeshNode extends NeuralNodeFrame {
  /** resolved shell position */
  x: number;
  y: number;
  z: number;
}

export interface MeshEdge {
  id: string;
  /** node indices */
  a: number;
  b: number;
  weight: number;
  /** quadratic bezier control point, bulged away from the core */
  cx: number;
  cy: number;
  cz: number;
}

export interface Pulse {
  /** index into `edges` */
  edge: number;
  /** 0..1 progress along the edge */
  t: number;
  speed: number;
  intensity: number;
}

export interface NeuralState {
  /** bumped whenever nodes/edges are replaced, so geometry can rebuild */
  version: number;
  nodes: MeshNode[];
  edges: MeshEdge[];
  /** decayed activation per node, index-aligned with `nodes` */
  levels: Float32Array;
  /** decayed signal strength per edge, index-aligned with `edges` */
  flows: Float32Array;
  /** per-region peak activation, for the HUD meters */
  regions: Record<string, number>;
  pulses: Pulse[];
  /** cumulative spike odometer reported by the backend */
  fired: number;
  /** performance.now() of the last non-empty activation frame */
  lastActivityAt: number;
  /** overall 0..1 "how awake is it" summary, smoothed in the render loop */
  arousal: number;
}

export const REGION_ORDER = [
  "sensory",
  "intake",
  "memory",
  "cortex",
  "effector",
  "motor",
] as const;

export type Region = (typeof REGION_ORDER)[number];

/** Region palette, shared by the 3D mesh and the 2D cortex map. */
export const REGION_COLOR: Record<string, string> = {
  sensory: "#5eb0ff",
  intake: "#7ee0d0",
  memory: "#c9a6ff",
  cortex: "#eaf6ff",
  effector: "#ffc861",
  motor: "#4ade80",
};

export const neural: NeuralState = {
  version: 0,
  nodes: [],
  edges: [],
  levels: new Float32Array(0),
  flows: new Float32Array(0),
  regions: {},
  pulses: [],
  fired: 0,
  lastActivityAt: 0,
  arousal: 0,
};

/** Place a node on its layer ring. See the SHELL_* notes above for why. */
function place(layer: number, index: number, count: number, layers: number): [number, number, number] {
  const span = Math.max(1, layers - 1);
  const t = layer / span; // 0 at the top layer, 1 at the bottom
  const y = (0.5 - t) * 2 * SHELL_HEIGHT;
  // Widest at the equator, so the stack reads as a barrel rather than a tube.
  const r = SHELL_RADIUS * (1 - SHELL_BULGE + SHELL_BULGE * Math.sin(Math.PI * t));
  // Layers share a phase so that node i of one layer sits near node i of the
  // next, which is what keeps synapses short and the flow readable. The half
  // step centres each ring's first node on the front of the barrel.
  const theta = ((index + 0.5) / Math.max(1, count)) * Math.PI * 2;
  return [r * Math.cos(theta), y, r * Math.sin(theta)];
}

/** Replace the topology from a `neural.graph` frame and re-run layout. */
export function setGraph(nodes: NeuralNodeFrame[], edges: NeuralEdgeFrame[]): void {
  const layers = Math.max(1, ...nodes.map((n) => n.layer + 1));
  const perLayer = new Map<number, NeuralNodeFrame[]>();
  for (const node of nodes) {
    const bucket = perLayer.get(node.layer);
    if (bucket) bucket.push(node);
    else perLayer.set(node.layer, [node]);
  }

  const index = new Map<string, number>();
  const placed: MeshNode[] = nodes.map((node) => {
    const siblings = perLayer.get(node.layer)!;
    const [x, y, z] = place(node.layer, siblings.indexOf(node), siblings.length, layers);
    return { ...node, x, y, z };
  });
  placed.forEach((node, i) => index.set(node.id, i));

  const meshEdges: MeshEdge[] = [];
  for (const edge of edges) {
    const a = index.get(edge.from);
    const b = index.get(edge.to);
    if (a === undefined || b === undefined) continue;
    const na = placed[a];
    const nb = placed[b];
    const mx = (na.x + nb.x) / 2;
    const my = (na.y + nb.y) / 2;
    const mz = (na.z + nb.z) / 2;
    // Push the control point out along the barrel's *radial* axis (x/z only),
    // leaving height alone. Normalising against the origin instead would send
    // arcs wandering, because two nodes on opposite sides of a ring have a
    // midpoint sitting almost exactly on the axis where the direction is
    // undefined.
    const radial = Math.hypot(mx, mz);
    const push = radial < 0.001 ? 1 : (radial + SHELL_RADIUS * 0.2) / radial;
    meshEdges.push({
      id: edge.id,
      a,
      b,
      weight: edge.weight,
      cx: mx * push,
      cy: my,
      cz: mz * push,
    });
  }

  neural.nodes = placed;
  neural.edges = meshEdges;
  neural.levels = new Float32Array(placed.length);
  neural.flows = new Float32Array(meshEdges.length);
  neural.pulses = [];
  neural.version += 1;
}

/** Apply one coalesced `neural` activation frame. */
export function applyActivation(
  levels: number[],
  flows: Array<[number, number]>,
  regions: Record<string, number>,
  fired: number,
): void {
  const target = neural.levels;
  const count = Math.min(target.length, levels.length);
  for (let i = 0; i < count; i++) target[i] = levels[i];

  for (const [edgeIndex, intensity] of flows) {
    if (edgeIndex < 0 || edgeIndex >= neural.flows.length) continue;
    neural.flows[edgeIndex] = Math.min(1, neural.flows[edgeIndex] + intensity);
    spawnPulse(edgeIndex, intensity);
  }

  neural.regions = regions;
  neural.fired = fired;
  neural.lastActivityAt = performance.now();
}

/** Launch one action potential down an edge. */
export function spawnPulse(edge: number, intensity: number): void {
  if (neural.pulses.length >= MAX_PULSES) {
    // Drop the oldest rather than refusing the newest: a burst of fresh
    // activity is what the viewer is actually looking at.
    neural.pulses.shift();
  }
  neural.pulses.push({
    edge,
    t: 0,
    speed: 0.9 + intensity * 1.5 + Math.random() * 0.35,
    intensity: Math.max(0.25, Math.min(1, intensity)),
  });
}

/**
 * Advance pulses and decay edge activation. Called once per rendered frame by
 * whichever component owns the loop, so the 2D and 3D views stay in step.
 */
export function stepNeural(dt: number): void {
  const decay = Math.max(0, 1 - FLOW_DECAY * dt);
  for (let i = 0; i < neural.flows.length; i++) {
    const value = neural.flows[i] * decay;
    neural.flows[i] = value > 0.004 ? value : 0;
  }

  const alive: Pulse[] = [];
  for (const pulse of neural.pulses) {
    pulse.t += pulse.speed * dt;
    if (pulse.t < 1) alive.push(pulse);
  }
  neural.pulses = alive;

  let peak = 0;
  for (let i = 0; i < neural.levels.length; i++) {
    if (neural.levels[i] > peak) peak = neural.levels[i];
  }
  neural.arousal += (peak - neural.arousal) * Math.min(1, dt * 6);
}

/** Position along an edge's bezier at progress `t`, written into `out`. */
export function pointOnEdge(edge: MeshEdge, t: number, out: [number, number, number]): void {
  const a = neural.nodes[edge.a];
  const b = neural.nodes[edge.b];
  if (!a || !b) return;
  const inv = 1 - t;
  const w0 = inv * inv;
  const w1 = 2 * inv * t;
  const w2 = t * t;
  out[0] = w0 * a.x + w1 * edge.cx + w2 * b.x;
  out[1] = w0 * a.y + w1 * edge.cy + w2 * b.y;
  out[2] = w0 * a.z + w1 * edge.cz + w2 * b.z;
}
