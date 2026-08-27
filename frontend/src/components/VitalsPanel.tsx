"use client";

/**
 * Left-edge system vitals: real host metrics streamed from the backend's
 * `vitals` frames (psutil-backed), rendered as thin arc gauges with a rolling
 * sparkline history. Nothing here is simulated.
 */

import { useEffect, useRef } from "react";
import { useJarvis } from "@/state/jarvis";

const HISTORY = 48;

/** A 270° arc gauge. */
function Gauge({
  label,
  value,
  display,
  tone = "blue",
}: {
  label: string;
  value: number;
  display: string;
  tone?: "blue" | "amber" | "red" | "green";
}) {
  const pct = Math.max(0, Math.min(100, value)) / 100;
  const r = 22;
  const circumference = 2 * Math.PI * r;
  const arc = circumference * 0.75; // 270° sweep
  const filled = arc * pct;

  return (
    <div className="gauge" data-tone={tone}>
      <svg viewBox="0 0 56 56" className="gauge-svg" aria-hidden="true">
        <circle
          className="gauge-track"
          cx="28"
          cy="28"
          r={r}
          strokeDasharray={`${arc} ${circumference}`}
        />
        <circle
          className="gauge-fill"
          cx="28"
          cy="28"
          r={r}
          strokeDasharray={`${filled} ${circumference}`}
        />
      </svg>
      <div className="gauge-inner">
        <span className="gauge-value">{display}</span>
      </div>
      <span className="gauge-label">{label}</span>
    </div>
  );
}

/** Rolling sparkline of a single metric. */
function Spark({ series, label }: { series: number[]; label: string }) {
  const max = Math.max(1, ...series);
  const points = series
    .map((v, i) => {
      const x = (i / Math.max(1, HISTORY - 1)) * 100;
      const y = 24 - (v / max) * 22;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <div className="spark">
      <span className="spark-label">{label}</span>
      <svg viewBox="0 0 100 24" preserveAspectRatio="none" className="spark-svg" aria-hidden="true">
        <polyline points={points} />
      </svg>
    </div>
  );
}

export default function VitalsPanel() {
  const vitals = useJarvis((s) => s.vitals);
  const cpuHistory = useRef<number[]>(new Array(HISTORY).fill(0));
  const netHistory = useRef<number[]>(new Array(HISTORY).fill(0));

  useEffect(() => {
    if (!vitals) return;
    cpuHistory.current = [...cpuHistory.current.slice(1), vitals.cpu ?? 0];
    netHistory.current = [...netHistory.current.slice(1), vitals.net_kbps ?? 0];
  }, [vitals]);

  if (!vitals) {
    return (
      <aside className="vitals" aria-label="System vitals">
        <div className="vitals-title">SYSTEM VITALS</div>
        <div className="vitals-empty">sampling…</div>
      </aside>
    );
  }

  const cpuTone = vitals.cpu > 88 ? "red" : vitals.cpu > 65 ? "amber" : "blue";
  const memTone = vitals.mem > 90 ? "red" : vitals.mem > 75 ? "amber" : "blue";

  return (
    <aside className="vitals" aria-label="System vitals">
      <div className="vitals-title">SYSTEM VITALS</div>

      <div className="vitals-gauges">
        <Gauge label="CPU" value={vitals.cpu ?? 0} display={`${Math.round(vitals.cpu ?? 0)}%`} tone={cpuTone} />
        <Gauge label="MEM" value={vitals.mem ?? 0} display={`${Math.round(vitals.mem ?? 0)}%`} tone={memTone} />
        <Gauge label="DISK" value={vitals.disk ?? 0} display={`${Math.round(vitals.disk ?? 0)}%`} tone="blue" />
      </div>

      <div className="vitals-rows">
        {vitals.mem_used_gb !== undefined && vitals.mem_total_gb !== undefined && (
          <div className="vitals-row">
            <span>memory</span>
            <b>
              {vitals.mem_used_gb.toFixed(1)} / {Math.round(vitals.mem_total_gb)} GB
            </b>
          </div>
        )}
        <div className="vitals-row">
          <span>load</span>
          <b>{(vitals.load ?? []).map((l) => l.toFixed(2)).join("  ")}</b>
        </div>
        <div className="vitals-row">
          <span>cores</span>
          <b>{vitals.cores}</b>
        </div>
        {vitals.procs !== undefined && (
          <div className="vitals-row">
            <span>processes</span>
            <b>{vitals.procs}</b>
          </div>
        )}
        {vitals.battery !== undefined && (
          <div className="vitals-row">
            <span>power</span>
            <b>
              {Math.round(vitals.battery)}% {vitals.power === "ac" ? "⚡" : "🔋"}
            </b>
          </div>
        )}
      </div>

      <Spark series={cpuHistory.current} label="cpu" />
      <Spark series={netHistory.current} label={`net ${Math.round(vitals.net_kbps ?? 0)} KB/s`} />
    </aside>
  );
}
