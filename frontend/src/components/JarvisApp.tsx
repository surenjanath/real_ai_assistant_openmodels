"use client";

/** Client root: composes the holographic scene and HUD layers. */

import dynamic from "next/dynamic";
import Console from "./Console";
import Hud from "./Hud";
import MathAnchor from "./MathAnchor";
import SettingsPanel from "./SettingsPanel";
import TelemetryPanel from "./TelemetryPanel";
import VitalsPanel from "./VitalsPanel";
import { useJarvisConnection } from "@/hooks/useJarvisConnection";
import { useJarvis } from "@/state/jarvis";

const Scene = dynamic(() => import("./Scene"), { ssr: false });

export default function JarvisApp() {
  useJarvisConnection();
  const status = useJarvis((s) => s.status);
  const logsConnected = useJarvis((s) => s.logsConnected);
  const wakeWordOn = useJarvis((s) => s.wakeWordOn);

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
      <VitalsPanel />
      <TelemetryPanel />
      <Console />
      <SettingsPanel />
      <MathAnchor />

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
