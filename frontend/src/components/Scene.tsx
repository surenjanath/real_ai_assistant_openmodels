"use client";

/**
 * The holographic core (PRD §2), tuned for maximum audio responsiveness:
 *
 *   - outer cage: dim gray 1px wireframe icosahedron, slow Y rotation
 *   - particle sphere: 4200 white glowing points on a fibonacci sphere with
 *     per-point phase attributes; every frame the CPU displaces points along
 *     their radial axes with travelling waves driven by level / bass / kick
 *   - FFT ring visualizer: 24 log-spaced spectrum bands mapped onto a
 *     mirrored 48-point ring orbiting the core
 *   - gyroscope rings: two tilted thin circles whose spin rate and opacity
 *     accelerate with bass energy
 *   - satellites: three bright dots orbiting the core, speed with bass
 *   - layered glow sprites: core + halo, intensity follows loudness
 *
 * All animation state lives in the mutable `audioLevels` singleton - no React
 * state is touched from the render loop.
 */

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { AdaptiveDpr } from "@react-three/drei";
import * as THREE from "three";
import { audioEngine } from "@/audio/engine";
import { audioLevels, BAND_COUNT } from "@/audio/levels";

const TAU = Math.PI * 2;

/* ---------- helpers ---------- */

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

function circlePoints(segments: number, radius: number): Float32Array {
  const positions = new Float32Array(segments * 3);
  for (let i = 0; i < segments; i++) {
    const theta = (i / segments) * TAU;
    positions[i * 3] = Math.cos(theta) * radius;
    positions[i * 3 + 1] = 0;
    positions[i * 3 + 2] = Math.sin(theta) * radius;
  }
  return positions;
}

/** Circle as segment PAIRS (v0-v1, v1-v2, ...) for <lineSegments>. */
function circleSegments(segments: number, radius: number): Float32Array {
  const positions = new Float32Array(segments * 2 * 3);
  for (let i = 0; i < segments; i++) {
    const a = (i / segments) * TAU;
    const b = ((i + 1) / segments) * TAU;
    positions[i * 6] = Math.cos(a) * radius;
    positions[i * 6 + 1] = 0;
    positions[i * 6 + 2] = Math.sin(a) * radius;
    positions[i * 6 + 3] = Math.cos(b) * radius;
    positions[i * 6 + 4] = 0;
    positions[i * 6 + 5] = Math.sin(b) * radius;
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
  return new THREE.CanvasTexture(canvas);
}

function dotTexture(): THREE.Texture {
  return radialTexture([
    [0, "rgba(255,255,255,1)"],
    [0.3, "rgba(255,255,255,0.6)"],
    [1, "rgba(255,255,255,0)"],
  ]);
}

function coreGlowTexture(): THREE.Texture {
  return radialTexture([
    [0, "rgba(190,220,255,0.85)"],
    [0.22, "rgba(130,175,255,0.4)"],
    [0.55, "rgba(70,115,220,0.16)"],
    [1, "rgba(30,60,140,0)"],
  ]);
}

function haloGlowTexture(): THREE.Texture {
  return radialTexture([
    [0, "rgba(90,140,255,0.30)"],
    [0.5, "rgba(50,90,200,0.12)"],
    [1, "rgba(20,45,120,0)"],
  ]);
}

/* ---------- outer cage ---------- */

function Cage() {
  const mesh = useRef<THREE.Mesh>(null);
  useFrame((_, dt) => {
    if (mesh.current) mesh.current.rotation.y += dt * 0.07; // slow continuous Y-axis spin
  });
  return (
    <mesh ref={mesh}>
      <icosahedronGeometry args={[2.42, 1]} />
      <meshBasicMaterial color="#4a515c" wireframe transparent opacity={0.4} />
    </mesh>
  );
}

/* ---------- particle sphere (displacement waves) ---------- */

const CORE_COUNT = 4200;
const CORE_RADIUS = 1.42;

function ParticleSphere() {
  const group = useRef<THREE.Group>(null);
  const material = useRef<THREE.PointsMaterial>(null);
  const haloMaterial = useRef<THREE.PointsMaterial>(null);
  const dot = useMemo(dotTexture, []);

  const { coreGeometry, base, phases, radii } = useMemo(() => {
    const positions = fibonacciSphere(CORE_COUNT, CORE_RADIUS, 0.05);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions.slice(), 3));
    const phaseA = new Float32Array(CORE_COUNT);
    const phaseB = new Float32Array(CORE_COUNT);
    const radii = new Float32Array(CORE_COUNT);
    for (let i = 0; i < CORE_COUNT; i++) {
      phaseA[i] = Math.random();
      phaseB[i] = Math.random();
      radii[i] = Math.hypot(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    }
    return {
      coreGeometry: geometry,
      base: positions,
      phases: { a: phaseA, b: phaseB },
      radii,
    };
  }, []);

  const haloGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(fibonacciSphere(600, 1.92, 0.02), 3));
    return geometry;
  }, []);

  const smooth = useRef({ scale: 1 });

  useFrame((state, dt) => {
    audioEngine.tick(dt, state.clock.elapsedTime);
    const { level, bass, treble, kick, speaking } = audioLevels;
    const t = state.clock.elapsedTime;

    // --- CPU displacement: travelling waves through the particle field ---
    const attr = coreGeometry.getAttribute("position") as THREE.BufferAttribute;
    const pos = attr.array as Float32Array;
    const waveAmp = 0.10 + level * 0.42 + kick * 0.18;
    const shimmer = 0.02 + treble * 0.10;
    for (let i = 0; i < CORE_COUNT; i++) {
      const r = radii[i];
      const wave =
        Math.sin(phases.a[i] * TAU + t * 2.4 + r * 4.2) * waveAmp +
        Math.sin(phases.b[i] * TAU - t * 3.3 + r * 2.0) * (0.35 * waveAmp) +
        Math.sin(t * 9.0 + phases.a[i] * TAU * 3.0) * shimmer;
      const factor = (r + wave) / r;
      pos[i * 3] = base[i * 3] * factor;
      pos[i * 3 + 1] = base[i * 3 + 1] * factor;
      pos[i * 3 + 2] = base[i * 3 + 2] * factor;
    }
    attr.needsUpdate = true;

    // --- group transform: pulse scale + accelerating rotation ---
    const targetScale = 1 + level * 0.34 + bass * 0.22 + kick * 0.3 + (speaking ? 0.04 : 0);
    smooth.current.scale += (targetScale - smooth.current.scale) * Math.min(1, dt * 11);
    if (group.current) {
      const s = smooth.current.scale;
      group.current.scale.set(s, s, s);
      group.current.rotation.y += dt * (0.1 + bass * 2.2 + level * 0.7 + kick * 0.8);
      group.current.rotation.x += dt * 0.018;
    }

    // --- material energy ---
    if (material.current) {
      material.current.size = 0.062 + level * 0.062 + treble * 0.03;
      material.current.opacity = Math.min(1, 0.55 + level * 0.5 + treble * 0.2);
    }
    if (haloMaterial.current) {
      haloMaterial.current.opacity = Math.min(0.9, 0.1 + treble * 0.45 + level * 0.25);
      haloMaterial.current.size = 0.14 + bass * 0.08;
    }
  });

  return (
    <group ref={group}>
      <points geometry={coreGeometry}>
        <pointsMaterial
          ref={material}
          map={dot}
          color="#ffffff"
          size={0.062}
          sizeAttenuation
          transparent
          opacity={0.62}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>
      <points geometry={haloGeometry}>
        <pointsMaterial
          ref={haloMaterial}
          map={dot}
          color="#9fc2ff"
          size={0.15}
          sizeAttenuation
          transparent
          opacity={0.14}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>
    </group>
  );
}

/* ---------- FFT ring visualizer ---------- */

const RING_POINTS = BAND_COUNT * 2;
const RING_RADIUS = 2.02;

function SpectrumRing() {
  const points = useRef<THREE.Points>(null);
  const material = useRef<THREE.PointsMaterial>(null);
  const dot = useMemo(dotTexture, []);

  const geometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(RING_POINTS * 3), 3));
    return geometry;
  }, []);

  useFrame((state) => {
    const attr = geometry.getAttribute("position") as THREE.BufferAttribute;
    const pos = attr.array as Float32Array;
    const t = state.clock.elapsedTime;
    const ringPulse = 1 + audioLevels.level * 0.16 + audioLevels.kick * 0.2;
    for (let i = 0; i < RING_POINTS; i++) {
      // Mirror the 24 bands for a symmetric profile.
      const band = i < BAND_COUNT ? i : RING_POINTS - 1 - i;
      const value = audioLevels.bands[band] ?? 0;
      const theta = (i / RING_POINTS) * TAU - Math.PI / 2;
      const radius = (RING_RADIUS + value * 0.55) * ringPulse;
      const lift = Math.sin(theta * 2 + t * 0.6) * 0.02;
      pos[i * 3] = Math.cos(theta) * radius;
      pos[i * 3 + 1] = lift;
      pos[i * 3 + 2] = Math.sin(theta) * radius;
    }
    attr.needsUpdate = true;
    if (points.current) {
      points.current.rotation.y = t * 0.22;
    }
    if (material.current) {
      material.current.opacity = 0.35 + audioLevels.level * 0.55;
      material.current.size = 0.045 + audioLevels.treble * 0.035;
    }
  });

  return (
    <points ref={points} geometry={geometry}>
      <pointsMaterial
        ref={material}
        map={dot}
        color="#8fd0ff"
        size={0.05}
        sizeAttenuation
        transparent
        opacity={0.4}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

/* ---------- gyroscope rings ---------- */

function GyroRings() {
  const inner = useRef<THREE.LineSegments>(null);
  const outer = useRef<THREE.LineSegments>(null);
  const innerMat = useRef<THREE.LineBasicMaterial>(null);
  const outerMat = useRef<THREE.LineBasicMaterial>(null);

  const innerGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(circleSegments(160, 2.16), 3));
    return geometry;
  }, []);
  const outerGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(circleSegments(160, 2.3), 3));
    return geometry;
  }, []);

  useFrame((state, dt) => {
    const t = state.clock.elapsedTime;
    const spin = 1 + audioLevels.bass * 4.5 + audioLevels.level * 2.0;
    if (inner.current) {
      inner.current.rotation.y += dt * 0.55 * spin;
      inner.current.rotation.z = Math.PI / 3 + Math.sin(t * 0.35) * 0.12;
    }
    if (outer.current) {
      outer.current.rotation.y -= dt * 0.38 * spin;
      outer.current.rotation.x = Math.PI / 2.6 + Math.sin(t * 0.27 + 1.7) * 0.15;
    }
    const glow = 0.08 + audioLevels.level * 0.4 + audioLevels.treble * 0.2;
    if (innerMat.current) innerMat.current.opacity = Math.min(0.7, glow);
    if (outerMat.current) outerMat.current.opacity = Math.min(0.55, glow * 0.7);
  });

  return (
    <group>
      <lineSegments ref={inner} geometry={innerGeometry}>
        <lineBasicMaterial ref={innerMat} color="#7fb4ff" transparent opacity={0.12} blending={THREE.AdditiveBlending} />
      </lineSegments>
      <lineSegments ref={outer} geometry={outerGeometry}>
        <lineBasicMaterial ref={outerMat} color="#5e8fd6" transparent opacity={0.08} blending={THREE.AdditiveBlending} />
      </lineSegments>
    </group>
  );
}

/* ---------- orbiting satellites ---------- */

const SATELLITES = [
  { radius: 2.62, speed: 0.9, tilt: 0.22, phase: 0.0, size: 0.16 },
  { radius: 2.78, speed: -0.62, tilt: -0.55, phase: 2.1, size: 0.12 },
  { radius: 2.5, speed: 1.28, tilt: 1.05, phase: 4.2, size: 0.1 },
];

function Satellites() {
  const group = useRef<THREE.Group>(null);
  const dot = useMemo(dotTexture, []);
  const refs = [useRef<THREE.Sprite>(null), useRef<THREE.Sprite>(null), useRef<THREE.Sprite>(null)];

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    const boost = 1 + audioLevels.bass * 2.6 + audioLevels.level * 1.4;
    SATELLITES.forEach((orbit, i) => {
      const sprite = refs[i].current;
      if (!sprite) return;
      const angle = orbit.phase + t * orbit.speed * boost;
      sprite.position.set(
        Math.cos(angle) * orbit.radius,
        Math.sin(angle) * orbit.radius * Math.sin(orbit.tilt),
        Math.sin(angle) * orbit.radius * Math.cos(orbit.tilt),
      );
      const pulse = orbit.size * (1 + audioLevels.treble * 0.8 + audioLevels.level * 0.5);
      sprite.scale.set(pulse, pulse, 1);
      (sprite.material as THREE.SpriteMaterial).opacity = 0.5 + audioLevels.level * 0.5;
    });
    if (group.current) group.current.rotation.y = t * 0.05;
  });

  return (
    <group ref={group}>
      {SATELLITES.map((_, i) => (
        <sprite key={i} ref={refs[i]}>
          <spriteMaterial
            map={dot}
            color={i === 0 ? "#cfe6ff" : "#9fc2ff"}
            transparent
            opacity={0.6}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
          />
        </sprite>
      ))}
    </group>
  );
}

/* ---------- layered glow ---------- */

function CoreGlow() {
  const coreMat = useRef<THREE.SpriteMaterial>(null);
  const haloMat = useRef<THREE.SpriteMaterial>(null);
  const core = useRef<THREE.Sprite>(null);
  const halo = useRef<THREE.Sprite>(null);
  const coreTexture = useMemo(coreGlowTexture, []);
  const haloTexture = useMemo(haloGlowTexture, []);

  useFrame((_, dt) => {
    const { level, kick } = audioLevels;
    if (coreMat.current) {
      const target = Math.min(0.9, 0.16 + level * 0.9 + kick * 0.25);
      coreMat.current.opacity += (target - coreMat.current.opacity) * Math.min(1, dt * 9);
    }
    if (haloMat.current) {
      const target = Math.min(0.55, 0.06 + level * 0.5);
      haloMat.current.opacity += (target - haloMat.current.opacity) * Math.min(1, dt * 6);
    }
    if (core.current) {
      const s = 5.6 + level * 2.6 + kick * 1.2;
      core.current.scale.set(s, s, 1);
    }
    if (halo.current) {
      const s = 9.5 + level * 3.5;
      halo.current.scale.set(s, s, 1);
    }
  });

  return (
    <group>
      <sprite ref={core}>
        <spriteMaterial ref={coreMat} map={coreTexture} transparent opacity={0.2} depthWrite={false} blending={THREE.AdditiveBlending} />
      </sprite>
      <sprite ref={halo}>
        <spriteMaterial ref={haloMat} map={haloTexture} transparent opacity={0.08} depthWrite={false} blending={THREE.AdditiveBlending} />
      </sprite>
    </group>
  );
}

/* ---------- scene root ---------- */

export default function Scene() {
  return (
    <Canvas
      className="scene-canvas"
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      camera={{ fov: 44, position: [0, 0.15, 6.6], near: 0.1, far: 50 }}
    >
      <AdaptiveDpr pixelated={false} />
      <Cage />
      <ParticleSphere />
      <SpectrumRing />
      <GyroRings />
      <Satellites />
      <CoreGlow />
    </Canvas>
  );
}
