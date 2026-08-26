"use client";

/** Client root: composes the holographic scene and HUD layers. */

import dynamic from "next/dynamic";
import Hud from "./Hud";
import MathAnchor from "./MathAnchor";
import TelemetryPanel from "./TelemetryPanel";
import { useJarvisConnection } from "@/hooks/useJarvisConnection";
import { useJarvis } from "@/state/jarvis";

const Scene = dynamic(() => import("./Scene"), { ssr: false });

export default function JarvisApp() {
  useJarvisConnection();
  const status = useJarvis((s) => s.status);
  const logsConnected = useJarvis((s) => s.logsConnected);

  return (
    <main className={`app ${status !== "boot" ? "ready" : ""}`}>
      <div className="scene-layer">
        <Scene />
      </div>
      <Hud />
      <TelemetryPanel />
      <MathAnchor />
      <div className={`boot-veil ${logsConnected ? "gone" : ""}`}>
        <span>ESTABLISHING UPLINK</span>
      </div>
    </main>
  );
}
