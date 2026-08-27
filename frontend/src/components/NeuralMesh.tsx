"use client";

/**
 * The cortical shell: the backend's cognitive graph, rendered as a living
 * neural network wrapped around the reactor.
 *
 *   - synapses    every edge drawn as a bezier arc bulging over the shell,
 *                 brightening in proportion to how much signal it just carried
 *   - somas       one glowing point per node, sized and coloured by its live
 *                 activation and its region
 *   - potentials  discrete pulses travelling node-to-node along the arcs, one
 *                 spawned per real hand-off inside the backend
 *
 * An enclosing wireframe hull was tried and removed: at any radius large
 * enough to contain the barrel, a low-detail icosahedron draws long straight
 * chords clean across the viewport, which read as scratches over the reactor.
 * The existing scene cage already supplies the sense of an enclosing volume.
 *
 * Nothing here is decorative motion. A pulse exists because a function in
 * `orchestrator.py` actually called the next stage; a node is bright because
 * that subsystem is doing work right now. Watching this while asking a
 * question shows the signal travel sensory → intake → memory → cortex →
 * tools → motor in real time.
 *
 * All per-frame state is read from the `neural` singleton, so the 20 Hz
 * activation stream never triggers a React render.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { audioLevels } from "@/audio/levels";
import {
  MAX_PULSES,
  REGION_COLOR,
  neural,
  pointOnEdge,
  stepNeural,
} from "@/state/neural";

/** Bezier samples per synapse. More is smoother and costs a draw-call nothing. */
const ARC_SEGMENTS = 12;

const scratch: [number, number, number] = [0, 0, 0];
const colorCache = new Map<string, THREE.Color>();

function regionColor(region: string): THREE.Color {
  let color = colorCache.get(region);
  if (!color) {
    color = new THREE.Color(REGION_COLOR[region] ?? "#8fd6ff");
    colorCache.set(region, color);
  }
  return color;
}

/* ------------------------------------------------------------------ somas -- */

const SOMA_VERT = /* glsl */ `
  uniform float uScale;
  uniform float uTime;
  attribute float aLevel;
  attribute float aBase;
  attribute vec3 aColor;
  varying float vLevel;
  varying vec3 vColor;

  void main() {
    vLevel = aLevel;
    vColor = aColor;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    // Idle nodes still breathe faintly, so a resting network reads as alive
    // rather than switched off.
    float idle = 0.5 + 0.5 * sin(uTime * 1.4 + aBase * 12.0);
    float size = aBase * (1.0 + aLevel * 2.6 + idle * 0.12);
    gl_PointSize = size * (uScale / -mv.z);
  }
`;

const SOMA_FRAG = /* glsl */ `
  varying float vLevel;
  varying vec3 vColor;

  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv);
    if (d > 0.5) discard;
    // A bright core inside a soft corona: the corona is what grows when the
    // node fires, which reads as excitation rather than as a bigger dot.
    float core = smoothstep(0.22, 0.0, d);
    float corona = smoothstep(0.5, 0.12, d);
    float alpha = core * 0.9 + corona * (0.18 + vLevel * 0.55);
    vec3 color = mix(vColor, vec3(1.0), core * (0.35 + vLevel * 0.5));
    gl_FragColor = vec4(color, alpha * (0.42 + vLevel * 0.58));
  }
`;

function Somas({ version }: { version: number }) {
  const material = useRef<THREE.ShaderMaterial>(null);

  const geometry = useMemo(() => {
    const nodes = neural.nodes;
    const positions = new Float32Array(nodes.length * 3);
    const levels = new Float32Array(nodes.length);
    const bases = new Float32Array(nodes.length);
    const colors = new Float32Array(nodes.length * 3);
    nodes.forEach((node, i) => {
      positions[i * 3] = node.x;
      positions[i * 3 + 1] = node.y;
      positions[i * 3 + 2] = node.z;
      // Tool nodes are grown at runtime and read as satellites of the bus, so
      // they sit visibly smaller than the standing architecture.
      bases[i] = node.kind === "tool" ? 0.055 : 0.085;
      const color = regionColor(node.region);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("aLevel", new THREE.BufferAttribute(levels, 1));
    geo.setAttribute("aBase", new THREE.BufferAttribute(bases, 1));
    geo.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
    return geo;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on topology
  }, [version]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  useFrame((state) => {
    const attr = geometry.getAttribute("aLevel") as THREE.BufferAttribute;
    const array = attr.array as Float32Array;
    const levels = neural.levels;
    const count = Math.min(array.length, levels.length);
    for (let i = 0; i < count; i++) array[i] = levels[i];
    attr.needsUpdate = true;

    if (material.current) {
      const cam = state.camera as THREE.PerspectiveCamera;
      const deviceHeight = state.size.height * state.viewport.dpr;
      material.current.uniforms.uScale.value =
        deviceHeight / (2 * Math.tan((cam.fov * Math.PI) / 360));
      material.current.uniforms.uTime.value = state.clock.elapsedTime;
    }
  });

  return (
    <points geometry={geometry} frustumCulled={false}>
      <shaderMaterial
        ref={material}
        vertexShader={SOMA_VERT}
        fragmentShader={SOMA_FRAG}
        uniforms={{ uScale: { value: 1146 }, uTime: { value: 0 } }}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

/* --------------------------------------------------------------- synapses -- */

function Synapses({ version }: { version: number }) {
  const geometry = useMemo(() => {
    const edges = neural.edges;
    const perEdge = ARC_SEGMENTS * 2; // segment pairs for <lineSegments>
    const positions = new Float32Array(edges.length * perEdge * 3);
    const colors = new Float32Array(edges.length * perEdge * 3);

    edges.forEach((edge, e) => {
      for (let s = 0; s < ARC_SEGMENTS; s++) {
        const t0 = s / ARC_SEGMENTS;
        const t1 = (s + 1) / ARC_SEGMENTS;
        const base = (e * perEdge + s * 2) * 3;
        pointOnEdge(edge, t0, scratch);
        positions[base] = scratch[0];
        positions[base + 1] = scratch[1];
        positions[base + 2] = scratch[2];
        pointOnEdge(edge, t1, scratch);
        positions[base + 3] = scratch[0];
        positions[base + 4] = scratch[1];
        positions[base + 5] = scratch[2];
      }
    });

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return geo;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on topology
  }, [version]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  useFrame(() => {
    const attr = geometry.getAttribute("color") as THREE.BufferAttribute;
    const colors = attr.array as Float32Array;
    const perEdge = ARC_SEGMENTS * 2;

    neural.edges.forEach((edge, e) => {
      const flow = neural.flows[e] ?? 0;
      const source = neural.nodes[edge.a];
      const tint = regionColor(source?.region ?? "cortex");
      // Resting synapses sit only just above black: bright enough that the
      // wiring is discernible when you look for it, dim enough that an idle
      // network does not read as scratches over the reactor. Carrying signal
      // is what makes one bloom toward white.
      const rest = 0.022 + edge.weight * 0.022;
      const level = rest + flow * 1.05;
      const white = flow * 0.6;
      const r = (tint.r * (1 - white) + white) * level;
      const g = (tint.g * (1 - white) + white) * level;
      const b = (tint.b * (1 - white) + white) * level;
      for (let v = 0; v < perEdge; v++) {
        const base = (e * perEdge + v) * 3;
        colors[base] = r;
        colors[base + 1] = g;
        colors[base + 2] = b;
      }
    });
    attr.needsUpdate = true;
  });

  return (
    <lineSegments geometry={geometry} frustumCulled={false}>
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={0.9}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </lineSegments>
  );
}

/* --------------------------------------------------------- action potentials */

const PULSE_VERT = /* glsl */ `
  uniform float uScale;
  attribute float aIntensity;
  varying float vIntensity;
  void main() {
    vIntensity = aIntensity;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = (0.05 + aIntensity * 0.075) * (uScale / -mv.z);
  }
`;

const PULSE_FRAG = /* glsl */ `
  varying float vIntensity;
  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv);
    if (d > 0.5) discard;
    float alpha = smoothstep(0.5, 0.0, d);
    alpha *= alpha;
    vec3 color = mix(vec3(0.55, 0.82, 1.0), vec3(1.0), vIntensity);
    gl_FragColor = vec4(color, alpha * vIntensity);
  }
`;

function Potentials() {
  const material = useRef<THREE.ShaderMaterial>(null);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAX_PULSES * 3), 3));
    geo.setAttribute("aIntensity", new THREE.BufferAttribute(new Float32Array(MAX_PULSES), 1));
    geo.setDrawRange(0, 0);
    return geo;
  }, []);

  useEffect(() => () => geometry.dispose(), [geometry]);

  useFrame((state, dt) => {
    // One step per frame for the whole neural layer, taken here because this
    // component always mounts with the mesh.
    stepNeural(Math.min(0.1, dt));

    const posAttr = geometry.getAttribute("position") as THREE.BufferAttribute;
    const intAttr = geometry.getAttribute("aIntensity") as THREE.BufferAttribute;
    const positions = posAttr.array as Float32Array;
    const intensities = intAttr.array as Float32Array;

    let written = 0;
    for (const pulse of neural.pulses) {
      if (written >= MAX_PULSES) break;
      const edge = neural.edges[pulse.edge];
      if (!edge) continue;
      pointOnEdge(edge, pulse.t, scratch);
      positions[written * 3] = scratch[0];
      positions[written * 3 + 1] = scratch[1];
      positions[written * 3 + 2] = scratch[2];
      // Fade in and out so a potential arrives and departs rather than
      // popping into and out of existence at the synapse endpoints.
      const envelope = Math.sin(Math.PI * Math.min(1, Math.max(0, pulse.t)));
      intensities[written] = pulse.intensity * (0.35 + envelope * 0.65);
      written++;
    }
    posAttr.needsUpdate = true;
    intAttr.needsUpdate = true;
    geometry.setDrawRange(0, written);

    if (material.current) {
      const cam = state.camera as THREE.PerspectiveCamera;
      const deviceHeight = state.size.height * state.viewport.dpr;
      material.current.uniforms.uScale.value =
        deviceHeight / (2 * Math.tan((cam.fov * Math.PI) / 360));
    }
  });

  return (
    <points geometry={geometry} frustumCulled={false}>
      <shaderMaterial
        ref={material}
        vertexShader={PULSE_VERT}
        fragmentShader={PULSE_FRAG}
        uniforms={{ uScale: { value: 1146 } }}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

/* ------------------------------------------------------------------- root -- */

export default function NeuralMesh() {
  // The graph arrives over the socket after mount, so watch the version
  // counter and rebuild geometry when the topology actually changes.
  const [version, setVersion] = useState(neural.version);
  const group = useRef<THREE.Group>(null);

  useEffect(() => {
    let raf = 0;
    const poll = () => {
      if (neural.version !== version) setVersion(neural.version);
      raf = requestAnimationFrame(poll);
    };
    raf = requestAnimationFrame(poll);
    return () => cancelAnimationFrame(raf);
  }, [version]);

  useFrame((state, dt) => {
    if (!group.current) return;
    const t = state.clock.elapsedTime;
    // The shell counter-rotates against the core so the two layers separate
    // visually, and leans gently with arousal.
    group.current.rotation.y -= dt * (0.026 + neural.arousal * 0.09);
    group.current.rotation.x = Math.sin(t * 0.09) * 0.07;
    const scale = 1 + neural.arousal * 0.035 + audioLevels.level * 0.015;
    group.current.scale.setScalar(scale);
  });

  if (neural.nodes.length === 0) return null;

  return (
    <group ref={group}>
      <Synapses version={version} />
      <Somas version={version} />
      <Potentials />
    </group>
  );
}
