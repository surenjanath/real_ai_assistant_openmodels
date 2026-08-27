"use client";

/**
 * Cortex meters: the readable counterpart to the 3D neural shell.
 *
 * The mesh shows you *that* the network is firing and roughly where; these six
 * region meters say which part of the architecture is doing the work — sensory
 * intake, memory, cortex, tools, motor — at a glance and in one small panel.
 *
 * The per-node detail deliberately lives in the neural overlay rather than
 * here: the left rail has around 200px to spare between the identity cluster
 * and the vitals panel, and twenty-odd node rows crammed into that became a
 * 50px scroll box showing three of them. Clicking the header opens the full
 * map with room to be legible.
 *
 * Like the mesh, this reads the mutable `neural` singleton inside its own
 * animation frame and writes straight to the DOM. React renders it only when
 * the *topology* changes, never when activation does; at 20 activation frames
 * a second, doing otherwise would re-render the panel 1,200 times a minute for
 * a bar that moved two pixels.
 */

import { useEffect, useRef, useState } from "react";
import { REGION_COLOR, REGION_ORDER, neural } from "@/state/neural";
import { useJarvis } from "@/state/jarvis";

function compact(value: number): string {
  if (value < 1000) return String(value);
  if (value < 1_000_000) return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)}k`;
  return `${(value / 1_000_000).toFixed(1)}M`;
}

export default function CortexPanel() {
  const [version, setVersion] = useState(neural.version);
  const setOverlay = useJarvis((s) => s.setOverlay);

  const regionRefs = useRef<Record<string, HTMLSpanElement | null>>({});
  const firedRef = useRef<HTMLSpanElement>(null);

  /* Topology changes are rare (boot, and each new tool node), so polling the
     version counter once a frame is cheaper than any subscription. */
  useEffect(() => {
    let raf = 0;
    const poll = () => {
      if (neural.version !== version) setVersion(neural.version);
      raf = requestAnimationFrame(poll);
    };
    raf = requestAnimationFrame(poll);
    return () => cancelAnimationFrame(raf);
  }, [version]);

  /* The activation loop: direct DOM writes, never setState. */
  useEffect(() => {
    let raf = 0;
    let lastFired = -1;
    const paint = () => {
      for (const region of REGION_ORDER) {
        const el = regionRefs.current[region];
        if (!el) continue;
        const level = neural.regions[region] ?? 0;
        el.style.transform = `scaleX(${Math.max(0.012, level)})`;
        el.style.opacity = String(0.35 + level * 0.65);
      }
      if (firedRef.current && neural.fired !== lastFired) {
        lastFired = neural.fired;
        firedRef.current.textContent = compact(neural.fired);
      }
      raf = requestAnimationFrame(paint);
    };
    raf = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(raf);
  }, []);

  if (neural.nodes.length === 0) return null;

  return (
    <section className="cortex" aria-label="Neural cortex activity">
      <header className="cortex-head">
        <button
          type="button"
          className="cortex-toggle"
          onClick={() => setOverlay("neural")}
          title="Open the full cognitive map"
        >
          <span className="cortex-title">NEURAL CORTEX</span>
          <span className="cortex-caret">▸</span>
        </button>
        <span className="cortex-fired" title="Total node activations since boot">
          <span ref={firedRef}>0</span> fired
        </span>
      </header>

      <div className="cortex-regions">
        {REGION_ORDER.map((region) => (
          <div className="cortex-region" key={region}>
            <span className="cortex-region-name">{region}</span>
            <span className="cortex-track">
              <span
                className="cortex-fill"
                ref={(el) => {
                  regionRefs.current[region] = el;
                }}
                style={{ background: REGION_COLOR[region] }}
              />
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
