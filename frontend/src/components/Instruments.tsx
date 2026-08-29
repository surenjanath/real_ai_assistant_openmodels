"use client";

/**
 * Instrument bar: what the assistant costs, measured rather than claimed.
 *
 * Time-to-first-word is the number that decides whether a voice assistant
 * feels alive or broken, so it leads. Everything here comes from the backend's
 * `metrics` frames (real timings around real model calls) except the spike
 * rate, which is derived locally from the neural stream — it is the one
 * reading that must update continuously rather than once per directive.
 *
 * A click opens the deeper metrics view.
 */

import { useEffect, useRef } from "react";
import { neural } from "@/state/neural";
import { useJarvis } from "@/state/jarvis";

function ms(value: number): string {
  if (!value) return "—";
  return value < 1000 ? `${Math.round(value)}ms` : `${(value / 1000).toFixed(1)}s`;
}

/** Live spike rate, sampled from the odometer rather than counted per frame. */
function useSpikeRate(ref: React.RefObject<HTMLSpanElement>) {
  useEffect(() => {
    let last = neural.fired;
    let lastAt = performance.now();
    let smoothed = 0;
    const timer = setInterval(() => {
      const now = performance.now();
      const dt = Math.max(0.05, (now - lastAt) / 1000);
      const rate = Math.max(0, (neural.fired - last) / dt);
      last = neural.fired;
      lastAt = now;
      // A little smoothing: the raw figure jitters hard between idle and a
      // streaming answer, and a number that flickers is a number nobody reads.
      smoothed += (rate - smoothed) * 0.45;
      if (ref.current) {
        ref.current.textContent = smoothed < 1 ? "0" : smoothed.toFixed(0);
      }
    }, 250);
    return () => clearInterval(timer);
  }, [ref]);
}

export default function Instruments() {
  const metrics = useJarvis((s) => s.metrics);
  const status = useJarvis((s) => s.status);
  const nodes = useJarvis((s) => s.neuralNodes);
  const setOverlay = useJarvis((s) => s.setOverlay);
  const spikeRef = useRef<HTMLSpanElement>(null);
  useSpikeRate(spikeRef);

  if (status === "boot") return null;

  const last = metrics?.last;
  const usedTools = last?.tools_used?.length ? last.tools_used.join(" · ") : null;

  return (
    <button
      type="button"
      className="instruments"
      onClick={() => setOverlay("metrics")}
      title="Open the performance view"
      aria-label="Performance instruments"
    >
      {/* Time to the first *spoken* word, which is the wait the operator
          actually experiences. Falls back to the first generated token when
          the last directive was answered without speech. */}
      <span className="instr" title="Time from the directive to the first spoken word">
        <span className="instr-label">FIRST WORD</span>
        <span className="instr-value">
          {ms(metrics?.ttfa_ms?.last || metrics?.ttft_ms.last || 0)}
        </span>
      </span>
      <span className="instr-sep" />
      {/* The usual explanation for a slow first word. The model re-reads the
          entire prompt before returning a single token, so this number *is*
          the wait — and with tool schemas attached it starts near 3,000
          before a word has been said. */}
      <span
        className="instr"
        title="Prompt tokens read before the last answer began — clear the context (⌘P) if this is climbing"
      >
        <span className="instr-label">CONTEXT</span>
        <span className="instr-value">
          {last?.prompt_tokens ? `${(last.prompt_tokens / 1000).toFixed(1)}k` : "—"}
        </span>
      </span>
      <span className="instr-sep" />
      <span className="instr">
        <span className="instr-label">TOK/S</span>
        <span className="instr-value">{metrics?.tok_s.last ? metrics.tok_s.last.toFixed(0) : "—"}</span>
      </span>
      <span className="instr-sep" />
      <span className="instr">
        <span className="instr-label">SPIKES/S</span>
        <span className="instr-value">
          <span ref={spikeRef}>0</span>
        </span>
      </span>
      <span className="instr-sep" />
      <span className="instr">
        <span className="instr-label">NODES</span>
        <span className="instr-value">{nodes || "—"}</span>
      </span>
      <span className="instr-sep" />
      <span className="instr">
        <span className="instr-label">DIRECTIVES</span>
        <span className="instr-value">{metrics?.commands ?? 0}</span>
      </span>
      {usedTools && (
        <>
          <span className="instr-sep" />
          <span className="instr wide">
            <span className="instr-label">LAST TOOLS</span>
            <span className="instr-value tools">{usedTools}</span>
          </span>
        </>
      )}
    </button>
  );
}
