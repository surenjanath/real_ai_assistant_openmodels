"use client";

/**
 * Secondary views that deserve room to breathe: the performance history and
 * the skill catalogue. Both open from the instrument bar or the command
 * palette, and both close on Escape or a click outside.
 *
 * The settings panel keeps its own component because the backend can open it
 * by voice; these two are interface-only.
 */

import { useEffect, useRef, useState } from "react";
import { sendSkill } from "@/hooks/useJarvisConnection";
import { REGION_COLOR, REGION_ORDER, neural } from "@/state/neural";
import { useJarvis } from "@/state/jarvis";
import type { MetricsFrame } from "@/lib/protocol";

function ms(value: number): string {
  if (!value) return "—";
  return value < 1000 ? `${Math.round(value)}ms` : `${(value / 1000).toFixed(2)}s`;
}

/* --------------------------------------------------------------- metrics -- */

/** Bar chart of the recent run history — the shape of the tail matters more
 *  than any single number, and a p95 alone hides a bimodal distribution. */
function History({ metrics }: { metrics: MetricsFrame }) {
  const runs = metrics.history ?? [];
  if (runs.length === 0) return <div className="overlay-empty">no runs recorded yet.</div>;
  const peak = Math.max(1, ...runs.map((r) => r.total_ms));

  return (
    <div className="hist">
      {runs.map((run, i) => {
        const total = Math.max(2, (run.total_ms / peak) * 100);
        const ttft = Math.max(1, (run.ttft_ms / peak) * 100);
        return (
          <span
            className={`hist-bar ${run.error ? "bad" : run.kind === "control" ? "control" : ""}`}
            key={i}
            style={{ height: `${total}%` }}
            title={`${run.kind} · first word ${ms(run.ttft_ms)} · total ${ms(run.total_ms)}${
              run.tok_s ? ` · ${run.tok_s} tok/s` : ""
            }`}
          >
            {/* The inner segment marks how much of the wait was before the
                first word — the part the user actually experiences as lag. */}
            <span className="hist-ttft" style={{ height: `${(ttft / total) * 100}%` }} />
          </span>
        );
      })}
    </div>
  );
}

function MetricsView() {
  const metrics = useJarvis((s) => s.metrics);
  if (!metrics) return <div className="overlay-empty">no measurements yet.</div>;

  const last = metrics.last;
  return (
    <>
      <div className="stat-grid">
        <div className="stat">
          <span className="stat-label">FIRST WORD p50</span>
          <span className="stat-value">{ms(metrics.ttft_ms.p50)}</span>
          <span className="stat-sub">p95 {ms(metrics.ttft_ms.p95)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">FULL ANSWER p50</span>
          <span className="stat-value">{ms(metrics.total_ms.p50)}</span>
          <span className="stat-sub">p95 {ms(metrics.total_ms.p95)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">THROUGHPUT</span>
          <span className="stat-value">{metrics.tok_s.avg.toFixed(0)}</span>
          <span className="stat-sub">tok/s · best {metrics.tok_s.best.toFixed(0)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">DIRECTIVES</span>
          <span className="stat-value">{metrics.commands}</span>
          <span className="stat-sub">
            {metrics.tool_calls} tool calls · {metrics.errors} failed
          </span>
        </div>
      </div>

      <div className="overlay-head">RECENT RUNS</div>
      <History metrics={metrics} />
      <div className="hist-legend">
        <span>
          <i className="swatch ttft" /> waiting for the first word
        </span>
        <span>
          <i className="swatch total" /> generating
        </span>
        <span>
          <i className="swatch control" /> control command
        </span>
      </div>

      {last && (
        <>
          <div className="overlay-head">LAST DIRECTIVE</div>
          <div className="overlay-kv">
            <span>text</span>
            <b>{last.text}</b>
            <span>route</span>
            <b>
              {last.kind} · {last.mode} · {last.model}
            </b>
            <span>first word</span>
            <b>{ms(last.ttft_ms)}</b>
            <span>composed</span>
            <b>{ms(last.total_ms)}</b>
            {last.voice_ms !== null && (
              <>
                <span>to speech</span>
                <b>{ms(last.voice_ms)}</b>
              </>
            )}
            <span>generated</span>
            <b>
              {last.chars} chars{last.tok_s ? ` · ${last.tok_s} tok/s` : ""}
            </b>
            {last.tools_used.length > 0 && (
              <>
                <span>tools</span>
                <b>{last.tools_used.join(", ")}</b>
              </>
            )}
          </div>
        </>
      )}
    </>
  );
}

/* ---------------------------------------------------------------- skills -- */

const DANGER_NOTE: Record<string, string> = {
  safe: "no side effects",
  reads_files: "reads files inside the permitted workspace",
  executes: "runs an allow-listed shell command",
  network: "makes a network request",
};

function SkillsView() {
  const skills = useJarvis((s) => s.skills);
  const settings = useJarvis((s) => s.settings);
  const tools = useJarvis((s) => s.tools);

  if (skills.length === 0) return <div className="overlay-empty">no skills registered.</div>;

  return (
    <>
      <p className="overlay-note">
        {settings.tools_active
          ? "Armed. The cortex calls these itself whenever a question needs real data."
          : settings.tools_supported
            ? "Available but disabled — enable skills in settings."
            : `${settings.model} does not advertise tool calling, so these run only when you invoke them.`}
      </p>

      <div className="skill-list">
        {skills.map((skill) => {
          const used = tools.filter((t) => t.name === skill.name).length;
          return (
            <div className="skill" key={skill.name} data-danger={skill.danger}>
              <span className="skill-head">
                <b>{skill.name.replace(/_/g, " ")}</b>
                <span className="skill-tags">
                  {skill.danger !== "safe" && <em className="skill-danger">{skill.danger}</em>}
                  {used > 0 && <em className="skill-used">{used}×</em>}
                </span>
              </span>
              <span className="skill-desc">{skill.description}</span>
              <span className="skill-foot">
                <span className="skill-params">
                  {skill.params.length ? skill.params.join(", ") : "no arguments"}
                  {skill.danger !== "safe" ? ` · ${DANGER_NOTE[skill.danger] ?? ""}` : ""}
                </span>
                {skill.params.length === 0 && (
                  <button
                    type="button"
                    className="skill-run"
                    onClick={() => sendSkill(skill.name)}
                    title="Run this now and show the result in the TOOLS tab"
                  >
                    run
                  </button>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </>
  );
}

/* ---------------------------------------------------------------- neural -- */

/**
 * The whole cognitive graph, grouped by region and named.
 *
 * Painted from its own animation frame straight into the DOM, for the same
 * reason as the rail meters: activation arrives twenty times a second and must
 * never pass through React.
 */
function NeuralView() {
  const [version] = useState(neural.version);
  const barRefs = useRef<Array<HTMLSpanElement | null>>([]);
  const rowRefs = useRef<Array<HTMLDivElement | null>>([]);

  useEffect(() => {
    let raf = 0;
    const paint = () => {
      const levels = neural.levels;
      for (let i = 0; i < barRefs.current.length; i++) {
        const level = levels[i] ?? 0;
        const bar = barRefs.current[i];
        const row = rowRefs.current[i];
        if (bar) bar.style.transform = `scaleX(${Math.max(0.015, level)})`;
        if (row) row.style.opacity = String(0.4 + Math.min(1, level) * 0.6);
      }
      raf = requestAnimationFrame(paint);
    };
    raf = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(raf);
  }, [version]);

  if (neural.nodes.length === 0) {
    return <div className="overlay-empty">the cognitive graph has not arrived yet.</div>;
  }

  const edgeCount = neural.edges.length;

  return (
    <>
      <p className="overlay-note">
        {neural.nodes.length} nodes and {edgeCount} synapses. A node brightens while that
        subsystem is doing work, and a tool grows its own node the first time it is really
        called — so this is a record of exercised capability, not a list of intentions.
      </p>

      {REGION_ORDER.map((region) => {
        const members = neural.nodes
          .map((node, index) => ({ node, index }))
          .filter((entry) => entry.node.region === region);
        if (members.length === 0) return null;
        return (
          <div className="neural-region" key={region}>
            <div className="neural-region-head">
              <i className="swatch" style={{ background: REGION_COLOR[region] }} />
              {region}
              <em>{members.length}</em>
            </div>
            {members.map(({ node, index }) => (
              <div
                className="neural-row"
                key={node.id}
                data-kind={node.kind}
                ref={(el) => {
                  rowRefs.current[index] = el;
                }}
              >
                <span className="neural-name">{node.label}</span>
                <span className="neural-id">{node.id}</span>
                <span className="neural-track">
                  <span
                    className="neural-fill"
                    ref={(el) => {
                      barRefs.current[index] = el;
                    }}
                    style={{ background: REGION_COLOR[region] }}
                  />
                </span>
              </div>
            ))}
          </div>
        );
      })}
    </>
  );
}

/* ------------------------------------------------------------------ root -- */

export default function Overlays() {
  const overlay = useJarvis((s) => s.overlay);
  const setOverlay = useJarvis((s) => s.setOverlay);

  useEffect(() => {
    if (overlay === "none" || overlay === "settings") return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOverlay("none");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [overlay, setOverlay]);

  if (overlay !== "metrics" && overlay !== "skills" && overlay !== "neural") return null;

  const title =
    overlay === "metrics"
      ? "PERFORMANCE"
      : overlay === "skills"
        ? "SKILL CATALOGUE"
        : "COGNITIVE MAP";

  return (
    <>
      <div className="settings-scrim" onClick={() => setOverlay("none")} />
      <section className="overlay-panel" role="dialog" aria-label={title}>
        <header className="telemetry-header">
          <span className="telemetry-title">{title}</span>
          <button
            className="settings-close"
            onClick={() => setOverlay("none")}
            aria-label="Close"
            type="button"
          >
            ✕
          </button>
        </header>
        <div className="overlay-body">
          {overlay === "metrics" && <MetricsView />}
          {overlay === "skills" && <SkillsView />}
          {overlay === "neural" && <NeuralView />}
        </div>
      </section>
    </>
  );
}
