"use client";

/** Client root: composes the holographic scene and HUD layers. */

import dynamic from "next/dynamic";
import { useEffect } from "react";
import CommandPalette from "./CommandPalette";
import Console from "./Console";
import CortexPanel from "./CortexPanel";
import Hud from "./Hud";
import Instruments from "./Instruments";
import MathAnchor from "./MathAnchor";
import Overlays from "./Overlays";
import SettingsPanel from "./SettingsPanel";
import TelemetryPanel from "./TelemetryPanel";
import Toasts from "./Toasts";
import VitalsPanel from "./VitalsPanel";
import { useJarvisConnection } from "@/hooks/useJarvisConnection";
import { THEMES, useJarvis, type Theme } from "@/state/jarvis";

const Scene = dynamic(() => import("./Scene"), { ssr: false });

export default function JarvisApp() {
  useJarvisConnection();
  const status = useJarvis((s) => s.status);
  const logsConnected = useJarvis((s) => s.logsConnected);
  const wakeWordOn = useJarvis((s) => s.wakeWordOn);
  const setTheme = useJarvis((s) => s.setTheme);

  /* Restore the saved colour scheme. Done here rather than in the store's
     initialiser because the server render has no localStorage, and reading it
     during hydration would mismatch. */
  useEffect(() => {
    let saved: string | null = null;
    try {
      saved = window.localStorage.getItem("jarvis.theme");
    } catch {
      // Storage blocked; the default scheme is perfectly usable.
    }
    if (saved && (THEMES as readonly string[]).includes(saved)) {
      setTheme(saved as Theme);
    }
  }, [setTheme]);

  return (
    <main className={`app ${status !== "boot" ? "ready" : ""}`} data-status={status}>
      <div className="scene-layer">
        <Scene />
      </div>

      {/* corner frame */}
      <div className="frame" aria-hidden="true">
        <span className="corner tl" />
        <span className="corner tr" />
        <span className="corner bl" />
        <span className="corner br" />
      </div>

      {/* a soft vignette that reacts to state */}
      <div className={`state-glow ${status}`} aria-hidden="true" />
      {wakeWordOn && <div className="wake-ring" aria-hidden="true" />}

      <Hud />
      <Instruments />

      {/* One rail, so the cortex map and the vitals can never overlap however
          tall the window is — the cortex simply takes what is left over. */}
      <div className="left-rail">
        <CortexPanel />
        <VitalsPanel />
      </div>
      <TelemetryPanel />
      <Console />
      <SettingsPanel />
      <Overlays />
      <CommandPalette />
      <MathAnchor />
      <Toasts />

      <div className={`boot-veil ${logsConnected ? "gone" : ""}`}>
        <span className="boot-mark">J.A.R.V.I.S.</span>
        <span className="boot-text">ESTABLISHING UPLINK</span>
        <span className="boot-bar">
          <i />
        </span>
      </div>
    </main>
  );
}
