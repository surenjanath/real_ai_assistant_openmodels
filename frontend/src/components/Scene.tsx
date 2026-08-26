"use client";

/**
 * The holographic core (PRD §2):
 *   - outer cage: dim gray 1px wireframe icosahedron, slow Y rotation
 *   - particle sphere: white glowing points on a dense fibonacci sphere,
 *     pulsing/scaling/spinning with Web Audio frequency data
 *   - soft core glow sprite whose intensity tracks overall loudness
 */

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { AdaptiveDpr } from "@react-three/drei";
import * as THREE from "three";
import { audioEngine } from "@/audio/engine";
import { audioLevels } from "@/audio/levels";

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

function softDotTexture(): THREE.Texture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.35, "rgba(255,255,255,0.55)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function glowTexture(): THREE.Texture {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, "rgba(150,190,255,0.55)");
  gradient.addColorStop(0.35, "rgba(90,130,220,0.22)");
  gradient.addColorStop(1, "rgba(40,70,140,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

/* ---------- outer cage ---------- */

function Cage() {
  const mesh = useRef<THREE.Mesh>(null);
  useFrame((_, dt) => {
    if (mesh.current) mesh.current.rotation.y += dt * 0.07; // slow continuous Y-axis spin
  });
  return (
    <mesh ref={mesh}>
      <icosahedronGeometry args={[2.35, 1]} />
      <meshBasicMaterial color="#4a515c" wireframe transparent opacity={0.42} />
    </mesh>
  );
}

/* ---------- particle sphere ---------- */

function ParticleSphere() {
  const group = useRef<THREE.Group>(null);
  const material = useRef<THREE.PointsMaterial>(null);
  const haloMaterial = useRef<THREE.PointsMaterial>(null);
  const dot = useMemo(softDotTexture, []);

  const coreGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(fibonacciSphere(2600, 1.5), 3));
    return geometry;
  }, []);
  const haloGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(fibonacciSphere(500, 1.95, 0.02), 3));
    return geometry;
  }, []);

  const smooth = useRef({ scale: 1 });

  useFrame((state, dt) => {
    audioEngine.tick(dt, state.clock.elapsedTime);
    const { level, bass, treble, speaking } = audioLevels;

    // Scale pulse: breathing idle -> audio-driven swell.
    const targetScale = 1 + level * 0.42 + bass * 0.22 + (speaking ? 0.05 : 0);
    smooth.current.scale += (targetScale - smooth.current.scale) * Math.min(1, dt * 10);
    if (group.current) {
      const s = smooth.current.scale;
      group.current.scale.set(s, s, s);
      // Rotation accelerates with low-frequency energy.
      group.current.rotation.y += dt * (0.09 + bass * 1.35 + level * 0.5);
      group.current.rotation.x += dt * 0.016;
    }
    // Glow intensity + point size track treble/overall loudness.
    if (material.current) {
      material.current.size = 0.07 + level * 0.055 + treble * 0.02;
      material.current.opacity = 0.55 + level * 0.4;
    }
    if (haloMaterial.current) {
      haloMaterial.current.opacity = 0.10 + treble * 0.5 * level;
    }
  });

  return (
    <group ref={group}>
      <points geometry={coreGeometry}>
        <pointsMaterial
          ref={material}
          map={dot}
          color="#ffffff"
          size={0.07}
          sizeAttenuation
          transparent
          opacity={0.6}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>
      <points geometry={haloGeometry}>
        <pointsMaterial
          ref={haloMaterial}
          map={dot}
          color="#9fc2ff"
          size={0.16}
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

/* ---------- core glow ---------- */

function CoreGlow() {
  const material = useRef<THREE.SpriteMaterial>(null);
  const sprite = useRef<THREE.Sprite>(null);
  const texture = useMemo(glowTexture, []);
  useFrame((_, dt) => {
    const target = 0.1 + audioLevels.level * 0.85;
    if (material.current) {
      material.current.opacity += (Math.min(0.75, target) - material.current.opacity) * Math.min(1, dt * 8);
    }
    if (sprite.current) {
      const s = 6.2 + audioLevels.level * 2.4;
      sprite.current.scale.set(s, s, 1);
    }
  });
  return (
    <sprite ref={sprite}>
      <spriteMaterial
        ref={material}
        map={texture}
        transparent
        opacity={0.15}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </sprite>
  );
}

/* ---------- scene root ---------- */

export default function Scene() {
  return (
    <Canvas
      className="scene-canvas"
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      camera={{ fov: 42, position: [0, 0, 6.4], near: 0.1, far: 50 }}
    >
      <AdaptiveDpr pixelated={false} />
      <Cage />
      <ParticleSphere />
      <CoreGlow />
    </Canvas>
  );
}
