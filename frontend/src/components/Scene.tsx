"use client";

/**
 * The holographic core (PRD §2) - a JARVIS arc-reactor built from layered,
 * audio-reactive geometry:
 *
 *   - starfield        far parallax dust, very dim
 *   - cage             dim wireframe icosahedron, slow Y spin
 *   - particle core    24k GPU-displaced points on a fibonacci sphere; the
 *                      travelling waves run in a vertex shader, so the CPU
 *                      does no per-point work at all
 *   - reactor          hot inner core + layered additive glow sprites
 *   - spectrum bars    64 radial bars driven by the FFT bands
 *   - wave ring        128-point oscilloscope of the time-domain signal
 *   - hud rings        tick-marked technical rings + counter-rotating dashed arcs
 *   - gyro rings       three tilted circles that accelerate with bass
 *   - sweep            radar sweep line that brightens while thinking
 *   - satellites       orbiting dots with motion trails
 *
 * All animation state is read from the mutable `audioLevels` singleton - no
 * React state is touched from the render loop, so telemetry traffic never
 * re-renders the WebGL tree.
 */

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { AdaptiveDpr } from "@react-three/drei";
import * as THREE from "three";
import { audioEngine } from "@/audio/engine";
import { audioLevels, BAND_COUNT, WAVE_COUNT } from "@/audio/levels";

const TAU = Math.PI * 2;

/* ---------- palette ---------- */

const C_CORE = new THREE.Color("#eaf6ff");
const C_HOT = new THREE.Color("#8fd6ff");
const C_RIM = new THREE.Color("#3f7fd8");

/* ---------- geometry helpers ---------- */

function fibonacciSphere(count: number, radius: number, jitter = 0.04): Float32Array {
  const positions = new Float32Array(count * 3);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = goldenAngle * i;
    const wobble = 1 + (Math.random() - 0.5) * jitter * 2;
    positions[i * 3] = Math.cos(theta) * r * radius * wobble;
    positions[i * 3 + 1] = y * radius * wobble;
    positions[i * 3 + 2] = Math.sin(theta) * r * radius * wobble;
  }
  return positions;
}

/** Circle as segment PAIRS (v0-v1, v1-v2, ...) for <lineSegments>.
 *  Built in the XY plane so it faces the camera and reads as a circle;
 *  anything that should look tilted rotates itself. */
function circleSegments(segments: number, radius: number): Float32Array {
  const positions = new Float32Array(segments * 2 * 3);
  for (let i = 0; i < segments; i++) {
    const a = (i / segments) * TAU;
    const b = ((i + 1) / segments) * TAU;
    positions[i * 6] = Math.cos(a) * radius;
    positions[i * 6 + 1] = Math.sin(a) * radius;
    positions[i * 6 + 3] = Math.cos(b) * radius;
    positions[i * 6 + 4] = Math.sin(b) * radius;
  }
  return positions;
}

/** Dashed arc ring: `count` ticks of `span` radians each, at `radius`. */
function tickRing(count: number, radius: number, span: number, inner = 0): Float32Array {
  const positions = new Float32Array(count * 2 * 3);
  for (let i = 0; i < count; i++) {
    const a = (i / count) * TAU;
    const b = a + span;
    const r0 = inner || radius;
    positions[i * 6] = Math.cos(a) * r0;
    positions[i * 6 + 1] = Math.sin(a) * r0;
    positions[i * 6 + 3] = Math.cos(b) * radius;
    positions[i * 6 + 4] = Math.sin(b) * radius;
  }
  return positions;
}

/** Radial ticks pointing outward, used as a measurement scale. */
function radialTicks(count: number, radius: number, length: number): Float32Array {
  const positions = new Float32Array(count * 2 * 3);
  for (let i = 0; i < count; i++) {
    const a = (i / count) * TAU;
    const long = i % 5 === 0 ? length * 2.1 : length;
    positions[i * 6] = Math.cos(a) * radius;
    positions[i * 6 + 1] = Math.sin(a) * radius;
    positions[i * 6 + 3] = Math.cos(a) * (radius + long);
    positions[i * 6 + 4] = Math.sin(a) * (radius + long);
  }
  return positions;
}

function radialTexture(stops: Array<[number, string]>): THREE.Texture {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  for (const [offset, color] of stops) gradient.addColorStop(offset, color);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

const dotTexture = () =>
  radialTexture([
    [0, "rgba(255,255,255,1)"],
    [0.25, "rgba(255,255,255,0.55)"],
    [1, "rgba(255,255,255,0)"],
  ]);

const coreGlowTexture = () =>
  radialTexture([
    [0, "rgba(225,244,255,0.95)"],
    [0.14, "rgba(150,205,255,0.55)"],
    [0.4, "rgba(70,130,235,0.18)"],
    [1, "rgba(20,50,130,0)"],
  ]);

const haloGlowTexture = () =>
  radialTexture([
    [0, "rgba(96,158,255,0.30)"],
    [0.45, "rgba(48,92,205,0.11)"],
    [1, "rgba(16,40,110,0)"],
  ]);

/* ---------- starfield ---------- */

function Starfield() {
  const dot = useMemo(dotTexture, []);
  const group = useRef<THREE.Points>(null);
  const geometry = useMemo(() => {
    const count = 520;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      // Shell well behind the core so it parallaxes rather than intersects.
      const r = 16 + Math.random() * 26;
      const theta = Math.random() * TAU;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.cos(phi) * 0.6;
      positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }, []);

  useFrame((_, dt) => {
    if (group.current) group.current.rotation.y += dt * 0.006;
  });

  return (
    <points ref={group} geometry={geometry}>
      <pointsMaterial
        map={dot}
        color="#7fa8dd"
        size={0.09}
        sizeAttenuation
        transparent
        opacity={0.22}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

/* ---------- outer cage ---------- */

function Cage() {
  const mesh = useRef<THREE.Mesh>(null);
  const material = useRef<THREE.MeshBasicMaterial>(null);
  useFrame((_, dt) => {
    if (mesh.current) mesh.current.rotation.y += dt * 0.07;
    if (material.current) {
      const target = 0.16 + audioLevels.level * 0.2 + audioLevels.thinking * 0.12;
      material.current.opacity += (target - material.current.opacity) * Math.min(1, dt * 5);
    }
  });
  return (
    <mesh ref={mesh}>
      <icosahedronGeometry args={[3.62, 0]} />
      <meshBasicMaterial ref={material} color="#5c6675" wireframe transparent opacity={0.3} />
    </mesh>
  );
}

/* ---------- particle core (GPU displacement) ---------- */

const CORE_COUNT = 24000;
const CORE_RADIUS = 1.22;

const CORE_VERT = /* glsl */ `
  uniform float uTime;
  uniform float uLevel;
  uniform float uBass;
  uniform float uTreble;
  uniform float uKick;
  uniform float uThinking;
  uniform float uSize;
  uniform float uScale;

  attribute float aPhaseA;
  attribute float aPhaseB;

  varying float vGlow;

  void main() {
    vec3 dir = normalize(position);
    float r = length(position);

    // Travelling displacement waves, exactly the CPU formula that used to run
    // per-point on the main thread - now free, and over 5x the point count.
    float amp = 0.055 + uLevel * 0.20 + uKick * 0.10;
    float shimmer = 0.012 + uTreble * 0.05;
    float wave =
      sin(aPhaseA * 6.2831853 + uTime * 2.4 + r * 4.2) * amp +
      sin(aPhaseB * 6.2831853 - uTime * 3.3 + r * 2.0) * (0.35 * amp) +
      sin(uTime * 9.0 + aPhaseA * 18.849556) * shimmer;

    // While thinking, a slow vertical scan band ripples through the sphere.
    float scan = sin(dir.y * 7.0 - uTime * 3.2) * uThinking * 0.07;
    wave += scan;

    vec3 displaced = dir * (r + wave);

    vGlow = clamp((wave / max(amp, 0.001)) * 0.5 + 0.5, 0.0, 1.0);

    vec4 mv = modelViewMatrix * vec4(displaced, 1.0);
    gl_Position = projectionMatrix * mv;
    float size = uSize * (1.0 + uLevel * 0.55 + uTreble * 0.3);
    gl_PointSize = size * (uScale / -mv.z);
  }
`;

const CORE_FRAG = /* glsl */ `
  uniform vec3 uColorHot;
  uniform vec3 uColorRim;
  uniform float uOpacity;

  varying float vGlow;

  void main() {
    // Soft round sprite without a texture fetch.
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv);
    if (d > 0.5) discard;
    float alpha = smoothstep(0.5, 0.0, d);
    alpha *= alpha;

    vec3 color = mix(uColorRim, uColorHot, vGlow);
    gl_FragColor = vec4(color, alpha * uOpacity);
  }
`;

function ParticleCore() {
  const group = useRef<THREE.Group>(null);
  const smooth = useRef({ scale: 1 });

  const { geometry, material } = useMemo(() => {
    const positions = fibonacciSphere(CORE_COUNT, CORE_RADIUS, 0.05);
    const phaseA = new Float32Array(CORE_COUNT);
    const phaseB = new Float32Array(CORE_COUNT);
    for (let i = 0; i < CORE_COUNT; i++) {
      phaseA[i] = Math.random();
      phaseB[i] = Math.random();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("aPhaseA", new THREE.BufferAttribute(phaseA, 1));
    geo.setAttribute("aPhaseB", new THREE.BufferAttribute(phaseB, 1));

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uLevel: { value: 0 },
        uBass: { value: 0 },
        uTreble: { value: 0 },
        uKick: { value: 0 },
        uThinking: { value: 0 },
        // uSize is a world-space diameter; uScale converts it to pixels.
        uSize: { value: 0.019 },
        uScale: { value: 1146 },
        uOpacity: { value: 0.55 },
        uColorHot: { value: C_CORE.clone() },
        uColorRim: { value: C_RIM.clone() },
      },
      vertexShader: CORE_VERT,
      fragmentShader: CORE_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    return { geometry: geo, material: mat };
  }, []);

  useFrame((state, dt) => {
    // One tick per frame, before anything reads the levels.
    audioEngine.tick(dt, state.clock.elapsedTime);
    const { level, bass, treble, kick, speaking, thinking } = audioLevels;

    const u = material.uniforms;
    u.uTime.value = state.clock.elapsedTime;
    u.uLevel.value = level;
    u.uBass.value = bass;
    u.uTreble.value = treble;
    u.uKick.value = kick;
    u.uThinking.value = thinking;
    u.uOpacity.value = Math.min(0.95, 0.52 + level * 0.4 + treble * 0.16);
    // Convert a world-space point diameter into DEVICE pixels for this
    // projection (gl_PointSize is device px, so the DPR must be folded in):
    //   px = worldSize * (deviceHeight / (2 tan(fov/2))) / distance
    const cam = state.camera as THREE.PerspectiveCamera;
    const deviceHeight = state.size.height * state.viewport.dpr;
    u.uScale.value = deviceHeight / (2 * Math.tan((cam.fov * Math.PI) / 360));

    const targetScale = 1 + level * 0.14 + bass * 0.09 + kick * 0.12 + (speaking ? 0.03 : 0);
    smooth.current.scale += (targetScale - smooth.current.scale) * Math.min(1, dt * 11);
    if (group.current) {
      const s = smooth.current.scale;
      group.current.scale.setScalar(s);
      group.current.rotation.y += dt * (0.09 + bass * 2.0 + level * 0.65 + kick * 0.8);
      group.current.rotation.x += dt * 0.016;
    }
  });

  return (
    <group ref={group}>
      <points geometry={geometry} material={material} frustumCulled={false} />
    </group>
  );
}

/* ---------- reactor core + glow ---------- */

function Reactor() {
  const coreMat = useRef<THREE.SpriteMaterial>(null);
  const haloMat = useRef<THREE.SpriteMaterial>(null);
  const core = useRef<THREE.Sprite>(null);
  const halo = useRef<THREE.Sprite>(null);
  const seed = useRef<THREE.Mesh>(null);
  const seedMat = useRef<THREE.MeshBasicMaterial>(null);
  const coreTexture = useMemo(coreGlowTexture, []);
  const haloTexture = useMemo(haloGlowTexture, []);

  useFrame((state, dt) => {
    const { level, kick, thinking } = audioLevels;
    const t = state.clock.elapsedTime;

    if (coreMat.current) {
      const target = Math.min(0.5, 0.10 + level * 0.42 + kick * 0.16 + thinking * 0.12);
      coreMat.current.opacity += (target - coreMat.current.opacity) * Math.min(1, dt * 9);
    }
    if (haloMat.current) {
      const target = Math.min(0.34, 0.05 + level * 0.26 + thinking * 0.09);
      haloMat.current.opacity += (target - haloMat.current.opacity) * Math.min(1, dt * 6);
    }
    if (core.current) {
      const s = 3.4 + level * 1.8 + kick * 0.9;
      core.current.scale.set(s, s, 1);
    }
    if (halo.current) {
      const s = 7.6 + level * 2.8;
      halo.current.scale.set(s, s, 1);
    }
    // The hot seed at dead centre: a small icosahedron that breathes.
    if (seed.current) {
      const s = 0.085 + level * 0.06 + kick * 0.05 + Math.sin(t * 2.2) * 0.005;
      seed.current.scale.setScalar(s);
      seed.current.rotation.y += dt * 0.9;
      seed.current.rotation.x += dt * 0.4;
    }
    if (seedMat.current) {
      seedMat.current.opacity = Math.min(0.85, 0.34 + level * 0.45);
    }
  });

  return (
    <group>
      <sprite ref={halo}>
        <spriteMaterial
          ref={haloMat}
          map={haloTexture}
          transparent
          opacity={0.08}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </sprite>
      <sprite ref={core}>
        <spriteMaterial
          ref={coreMat}
          map={coreTexture}
          transparent
          opacity={0.2}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </sprite>
      <mesh ref={seed}>
        <icosahedronGeometry args={[1, 1]} />
        <meshBasicMaterial
          ref={seedMat}
          color={C_CORE}
          transparent
          opacity={0.7}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

/* ---------- spectrum bars ---------- */

const BAR_RADIUS = 1.96;

function SpectrumBars() {
  const lines = useRef<THREE.LineSegments>(null);
  const material = useRef<THREE.LineBasicMaterial>(null);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BAND_COUNT * 2 * 3), 3));
    return geo;
  }, []);

  useFrame((state) => {
    const attr = geometry.getAttribute("position") as THREE.BufferAttribute;
    const pos = attr.array as Float32Array;
    const t = state.clock.elapsedTime;
    const pulse = 1 + audioLevels.level * 0.1 + audioLevels.kick * 0.14;

    for (let i = 0; i < BAND_COUNT; i++) {
      const value = audioLevels.bands[i] ?? 0;
      const a = (i / BAND_COUNT) * TAU;
      const inner = BAR_RADIUS * pulse;
      const outer = inner + 0.07 + value * 0.5;
      const lift = Math.sin(a * 3 + t * 0.7) * 0.015;
      pos[i * 6] = Math.cos(a) * inner;
      pos[i * 6 + 1] = Math.sin(a) * inner;
      pos[i * 6 + 2] = lift;
      pos[i * 6 + 3] = Math.cos(a) * outer;
      pos[i * 6 + 4] = Math.sin(a) * outer;
      pos[i * 6 + 5] = lift;
    }
    attr.needsUpdate = true;

    if (lines.current) lines.current.rotation.z = t * 0.12;
    if (material.current) material.current.opacity = 0.42 + audioLevels.level * 0.5;
  });

  return (
    <lineSegments ref={lines} geometry={geometry}>
      <lineBasicMaterial
        ref={material}
        color="#8fd6ff"
        transparent
        opacity={0.4}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </lineSegments>
  );
}

/* ---------- oscilloscope wave ring ---------- */

function WaveRing() {
  const loop = useRef<THREE.Line>(null);
  const material = useRef<THREE.LineBasicMaterial>(null);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    // +1 vertex closes the loop back onto the first point.
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array((WAVE_COUNT + 1) * 3), 3));
    return geo;
  }, []);

  useFrame((state) => {
    const attr = geometry.getAttribute("position") as THREE.BufferAttribute;
    const pos = attr.array as Float32Array;
    const base = 1.70;
    for (let i = 0; i <= WAVE_COUNT; i++) {
      const idx = i % WAVE_COUNT;
      const a = (idx / WAVE_COUNT) * TAU;
      const r = base + (audioLevels.wave[idx] ?? 0) * 0.42;
      pos[i * 3] = Math.cos(a) * r;
      pos[i * 3 + 1] = Math.sin(a) * r;
      pos[i * 3 + 2] = 0;
    }
    attr.needsUpdate = true;
    if (loop.current) {
      loop.current.rotation.z = -state.clock.elapsedTime * 0.18;
    }
    if (material.current) {
      material.current.opacity = 0.34 + audioLevels.level * 0.6;
    }
  });

  return (
    // @ts-expect-error - R3F maps <line> to THREE.Line, which collides with SVG's line typing
    <line ref={loop} geometry={geometry}>
      <lineBasicMaterial
        ref={material}
        color="#cfe9ff"
        transparent
        opacity={0.4}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </line>
  );
}

/* ---------- technical HUD rings ---------- */

function HudRings() {
  const ticks = useRef<THREE.LineSegments>(null);
  const dashOuter = useRef<THREE.LineSegments>(null);
  const dashInner = useRef<THREE.LineSegments>(null);
  const tickMat = useRef<THREE.LineBasicMaterial>(null);
  const outerMat = useRef<THREE.LineBasicMaterial>(null);
  const innerMat = useRef<THREE.LineBasicMaterial>(null);

  const tickGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(radialTicks(90, 2.74, 0.06), 3));
    return geo;
  }, []);
  const outerGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(tickRing(9, 3.02, 0.42), 3));
    return geo;
  }, []);
  const innerGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(tickRing(14, 2.56, 0.26), 3));
    return geo;
  }, []);

  useFrame((state, dt) => {
    const t = state.clock.elapsedTime;
    const { level, thinking } = audioLevels;
    if (ticks.current) ticks.current.rotation.z += dt * 0.05;
    if (dashOuter.current) dashOuter.current.rotation.z += dt * (0.22 + thinking * 0.9);
    if (dashInner.current) dashInner.current.rotation.z -= dt * (0.34 + thinking * 1.3);
    const glow = 0.3 + level * 0.35 + thinking * 0.22;
    if (tickMat.current) tickMat.current.opacity = Math.min(0.6, glow * 0.8);
    if (outerMat.current) outerMat.current.opacity = Math.min(0.8, glow + 0.06 + Math.sin(t * 1.6) * 0.03);
    if (innerMat.current) innerMat.current.opacity = Math.min(0.75, glow);
  });

  return (
    <group>
      <lineSegments ref={ticks} geometry={tickGeo}>
        <lineBasicMaterial ref={tickMat} color="#5f8fd0" transparent opacity={0.2} blending={THREE.AdditiveBlending} depthWrite={false} />
      </lineSegments>
      <lineSegments ref={dashOuter} geometry={outerGeo}>
        <lineBasicMaterial ref={outerMat} color="#7fb8ff" transparent opacity={0.25} blending={THREE.AdditiveBlending} depthWrite={false} />
      </lineSegments>
      <lineSegments ref={dashInner} geometry={innerGeo}>
        <lineBasicMaterial ref={innerMat} color="#9fd0ff" transparent opacity={0.2} blending={THREE.AdditiveBlending} depthWrite={false} />
      </lineSegments>
    </group>
  );
}

/* ---------- gyroscope rings ---------- */

function GyroRings() {
  const a = useRef<THREE.LineSegments>(null);
  const b = useRef<THREE.LineSegments>(null);
  const c = useRef<THREE.LineSegments>(null);
  const matA = useRef<THREE.LineBasicMaterial>(null);
  const matB = useRef<THREE.LineBasicMaterial>(null);
  const matC = useRef<THREE.LineBasicMaterial>(null);

  const geoA = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(circleSegments(180, 2.22), 3));
    return g;
  }, []);
  const geoB = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(circleSegments(180, 2.36), 3));
    return g;
  }, []);
  const geoC = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(circleSegments(180, 2.5), 3));
    return g;
  }, []);

  useFrame((state, dt) => {
    const t = state.clock.elapsedTime;
    const spin = 1 + audioLevels.bass * 4.2 + audioLevels.level * 1.8;
    // Each ring keeps a steep fixed tilt (so it reads as a 3D gyroscope
    // hoop rather than a flat disc) and spins about its own axis.
    if (a.current) {
      a.current.rotation.x = 1.15 + Math.sin(t * 0.35) * 0.1;
      a.current.rotation.y += dt * 0.5 * spin;
    }
    if (b.current) {
      b.current.rotation.y = 1.2 + Math.sin(t * 0.27 + 1.7) * 0.12;
      b.current.rotation.x -= dt * 0.36 * spin;
    }
    if (c.current) {
      c.current.rotation.x = 0.75 + Math.cos(t * 0.31 + 0.6) * 0.1;
      c.current.rotation.z += dt * 0.28 * spin;
    }
    const glow = 0.16 + audioLevels.level * 0.38 + audioLevels.treble * 0.18;
    if (matA.current) matA.current.opacity = Math.min(0.6, glow);
    if (matB.current) matB.current.opacity = Math.min(0.5, glow * 0.8);
    if (matC.current) matC.current.opacity = Math.min(0.4, glow * 0.62);
  });

  return (
    <group>
      <lineSegments ref={a} geometry={geoA}>
        <lineBasicMaterial ref={matA} color="#7fb4ff" transparent opacity={0.12} blending={THREE.AdditiveBlending} depthWrite={false} />
      </lineSegments>
      <lineSegments ref={b} geometry={geoB}>
        <lineBasicMaterial ref={matB} color="#5e8fd6" transparent opacity={0.08} blending={THREE.AdditiveBlending} depthWrite={false} />
      </lineSegments>
      <lineSegments ref={c} geometry={geoC}>
        <lineBasicMaterial ref={matC} color="#4a79c0" transparent opacity={0.06} blending={THREE.AdditiveBlending} depthWrite={false} />
      </lineSegments>
    </group>
  );
}

/* ---------- radar sweep ---------- */

function Sweep() {
  const group = useRef<THREE.Group>(null);
  const material = useRef<THREE.LineBasicMaterial>(null);

  const geometry = useMemo(() => {
    // A fan of rays whose brightness falls off behind the leading edge.
    const rays = 26;
    const positions = new Float32Array(rays * 2 * 3);
    for (let i = 0; i < rays; i++) {
      const a = -(i / rays) * 0.75;
      positions[i * 6] = Math.cos(a) * 0.5;
      positions[i * 6 + 1] = Math.sin(a) * 0.5;
      positions[i * 6 + 3] = Math.cos(a) * 2.7;
      positions[i * 6 + 4] = Math.sin(a) * 2.7;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }, []);

  useFrame((state, dt) => {
    if (group.current) {
      group.current.rotation.z += dt * (0.5 + audioLevels.thinking * 2.4);
    }
    if (material.current) {
      material.current.opacity = 0.05 + audioLevels.thinking * 0.2 + audioLevels.level * 0.07;
    }
  });

  return (
    <group ref={group}>
      <lineSegments geometry={geometry}>
        <lineBasicMaterial ref={material} color="#6fb0ff" transparent opacity={0.05} blending={THREE.AdditiveBlending} depthWrite={false} />
      </lineSegments>
    </group>
  );
}

/* ---------- orbiting satellites ---------- */

const SATELLITES = [
  { radius: 3.24, speed: 0.85, tilt: 0.22, phase: 0.0, size: 0.15 },
  { radius: 3.42, speed: -0.58, tilt: -0.55, phase: 2.1, size: 0.11 },
  { radius: 3.1, speed: 1.2, tilt: 1.05, phase: 4.2, size: 0.09 },
];
const TRAIL = 18;

function Satellites() {
  const group = useRef<THREE.Group>(null);
  const dot = useMemo(dotTexture, []);
  const s0 = useRef<THREE.Sprite>(null);
  const s1 = useRef<THREE.Sprite>(null);
  const s2 = useRef<THREE.Sprite>(null);
  const refs = [s0, s1, s2];

  // One trail geometry per satellite, updated as a rolling history.
  const trails = useMemo(
    () =>
      SATELLITES.map(() => {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
        return geo;
      }),
    [],
  );

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    const boost = 1 + audioLevels.bass * 2.2 + audioLevels.level * 1.2;

    SATELLITES.forEach((orbit, i) => {
      const sprite = refs[i].current;
      if (!sprite) return;
      const angle = orbit.phase + t * orbit.speed * boost;
      const x = Math.cos(angle) * orbit.radius;
      const y = Math.sin(angle) * orbit.radius * Math.sin(orbit.tilt);
      const z = Math.sin(angle) * orbit.radius * Math.cos(orbit.tilt);
      sprite.position.set(x, y, z);
      const pulse = orbit.size * (1 + audioLevels.treble * 0.8 + audioLevels.level * 0.5);
      sprite.scale.set(pulse, pulse, 1);
      (sprite.material as THREE.SpriteMaterial).opacity = 0.45 + audioLevels.level * 0.5;

      // Trail: sample the orbit backwards in time so it always tracks exactly.
      const attr = trails[i].getAttribute("position") as THREE.BufferAttribute;
      const pos = attr.array as Float32Array;
      for (let k = 0; k < TRAIL; k++) {
        const back = angle - k * 0.05;
        pos[k * 3] = Math.cos(back) * orbit.radius;
        pos[k * 3 + 1] = Math.sin(back) * orbit.radius * Math.sin(orbit.tilt);
        pos[k * 3 + 2] = Math.sin(back) * orbit.radius * Math.cos(orbit.tilt);
      }
      attr.needsUpdate = true;
    });

    if (group.current) group.current.rotation.y = t * 0.04;
  });

  return (
    <group ref={group}>
      {SATELLITES.map((_, i) => (
        <group key={i}>
          <sprite ref={refs[i]}>
            <spriteMaterial
              map={dot}
              color={i === 0 ? "#dcefff" : "#9fc9ff"}
              transparent
              opacity={0.6}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
            />
          </sprite>
          {/* @ts-expect-error - R3F <line> vs SVG line typing */}
          <line geometry={trails[i]}>
            <lineBasicMaterial
              color="#6fa6ee"
              transparent
              opacity={0.16}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
            />
          </line>
        </group>
      ))}
    </group>
  );
}

/* ---------- camera drift ---------- */

function CameraRig() {
  useFrame((state, dt) => {
    // A very slow parallax drift keeps the still frame from feeling dead,
    // plus a gentle push-in while speaking.
    const t = state.clock.elapsedTime;
    const cam = state.camera;
    const targetZ = 7.0 - audioLevels.level * 0.35 - audioLevels.kick * 0.2;
    cam.position.x += (Math.sin(t * 0.13) * 0.16 - cam.position.x) * Math.min(1, dt * 0.6);
    cam.position.y += (0.18 + Math.cos(t * 0.11) * 0.12 - cam.position.y) * Math.min(1, dt * 0.6);
    cam.position.z += (targetZ - cam.position.z) * Math.min(1, dt * 2.2);
    cam.lookAt(0, 0, 0);
  });
  return null;
}

/* ---------- scene root ---------- */

export default function Scene() {
  return (
    <Canvas
      className="scene-canvas"
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      camera={{ fov: 44, position: [0, 0.18, 7.0], near: 0.1, far: 80 }}
    >
      <AdaptiveDpr pixelated={false} />
      <CameraRig />
      <Starfield />
      <Cage />
      <ParticleCore />
      <Reactor />
      <WaveRing />
      <SpectrumBars />
      <HudRings />
      <GyroRings />
      <Sweep />
      <Satellites />
    </Canvas>
  );
}
