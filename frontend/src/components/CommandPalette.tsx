"use client";

/**
 * Command palette (⌘P / Ctrl+P).
 *
 * Everything the assistant can do is reachable by voice, but voice is a poor
 * way to *discover* a capability — you cannot skim it. The palette is the
 * readable index of the same surface: control phrases, personas, installed
 * models, voices, themes, and every registered skill, all fuzzy-searchable and
 * executed through exactly the same paths the spoken commands use.
 *
 * Skills run directly (there is no point routing "check the disk" through a
 * language model when the tool is right there); everything else is dispatched
 * as the natural-language directive it corresponds to, so the backend's intent
 * grammar stays the single source of truth for what a command means.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { sendCommand, sendSettings, sendSkill } from "@/hooks/useJarvisConnection";
import { THEMES, useJarvis, type Theme } from "@/state/jarvis";

interface Entry {
  id: string;
  label: string;
  hint: string;
  group: string;
  run: () => void;
}

/** Subsequence match, so "swmd" finds "switch model". */
function fuzzy(haystack: string, needle: string): number {
  if (!needle) return 1;
  const hay = haystack.toLowerCase();
  const cut = needle.toLowerCase();
  const direct = hay.indexOf(cut);
  if (direct >= 0) return 1000 - direct;
  let score = 0;
  let at = 0;
  for (const ch of cut) {
    const found = hay.indexOf(ch, at);
    if (found < 0) return 0;
    // Adjacent characters are worth more than scattered ones.
    score += found === at ? 3 : 1;
    at = found + 1;
  }
  return score;
}

export default function CommandPalette() {
  const open = useJarvis((s) => s.paletteOpen);
  const setOpen = useJarvis((s) => s.setPaletteOpen);
  const settings = useJarvis((s) => s.settings);
  const skills = useJarvis((s) => s.skills);
  const theme = useJarvis((s) => s.theme);
  const setTheme = useJarvis((s) => s.setTheme);
  const setOverlay = useJarvis((s) => s.setOverlay);

  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const entries = useMemo<Entry[]>(() => {
    const say = (text: string) => () => sendCommand(text, "text");
    const out: Entry[] = [
      { id: "status", label: "System status", hint: "spoken diagnostic", group: "system", run: say("status") },
      { id: "metrics", label: "Performance report", hint: "latency & throughput", group: "system", run: say("performance report") },
      { id: "help", label: "What can you do?", hint: "spoken help", group: "system", run: say("help") },
      { id: "memstats", label: "What do you remember?", hint: "memory statistics", group: "memory", run: say("memory status") },
      { id: "notes", label: "Read my notes", hint: "list saved notes", group: "memory", run: say("list my notes") },
      { id: "reminders", label: "List reminders", hint: "pending reminders", group: "memory", run: say("list my reminders") },
      { id: "newchat", label: "New conversation", hint: "clear working memory", group: "memory", run: say("new conversation") },
      { id: "clear", label: "Clear the log", hint: "wipe telemetry", group: "system", run: say("clear the log") },
      { id: "stop", label: "Stop speaking", hint: "barge-in · Esc", group: "voice", run: say("stop") },
      { id: "faster", label: "Speak faster", hint: `speed ${settings.speed.toFixed(2)}`, group: "voice", run: say("speak faster") },
      { id: "slower", label: "Speak slower", hint: `speed ${settings.speed.toFixed(2)}`, group: "voice", run: say("speak slower") },
      {
        id: "think",
        label: settings.think_active ? "Disable extended thinking" : "Enable extended thinking",
        hint: settings.think_supported ? "chain-of-thought" : "unsupported on this model",
        group: "reasoning",
        run: say(settings.think_active ? "answer faster" : "think harder"),
      },
      {
        id: "tools",
        label: settings.tools ? "Disable skills" : "Enable skills",
        hint: settings.tools_supported ? "native tool calling" : "unsupported on this model",
        group: "reasoning",
        run: () => sendSettings({ tools: !settings.tools }),
      },
      {
        id: "recall",
        label: settings.recall ? "Disable long-term recall" : "Enable long-term recall",
        hint: "cross-session memory",
        group: "memory",
        run: () => sendSettings({ recall: !settings.recall }),
      },
      { id: "panel-settings", label: "Open settings", hint: "model, voice, persona", group: "panels", run: () => setOverlay("settings") },
      { id: "panel-metrics", label: "Open performance view", hint: "timing history", group: "panels", run: () => setOverlay("metrics") },
      { id: "panel-skills", label: "Open skill catalogue", hint: "what it can actually do", group: "panels", run: () => setOverlay("skills") },
      { id: "panel-neural", label: "Open cognitive map", hint: "every node and synapse", group: "panels", run: () => setOverlay("neural") },
      { id: "panel-archive", label: "Open the archive", hint: "past conversations & audit trail", group: "panels", run: () => setOverlay("archive") },
      { id: "panel-audit", label: "Audit trail", hint: "every skill this machine has run", group: "panels", run: () => setOverlay("archive") },
      { id: "panel-help", label: "Reference card", hint: "shortcuts & phrases · ?", group: "panels", run: () => setOverlay("help") },
      { id: "export", label: "Export transcript", hint: "opens plain text", group: "panels", run: () => window.open("/api/export?fmt=text&limit=500", "_blank") },
    ];

    for (const persona of settings.personas ?? []) {
      out.push({
        id: `persona-${persona.key}`,
        label: `Persona: ${persona.label}`,
        hint: persona.blurb,
        group: "persona",
        run: () => sendSettings({ persona: persona.key }),
      });
    }

    for (const model of settings.installed ?? []) {
      out.push({
        id: `model-${model}`,
        label: `Model: ${model}`,
        hint: model === settings.model ? "current" : "installed",
        group: "model",
        run: () => sendSettings({ model }),
      });
    }

    for (const voice of settings.voices ?? []) {
      out.push({
        id: `voice-${voice}`,
        label: `Voice: ${settings.voice_labels?.[voice] ?? voice}`,
        hint: voice === settings.voice ? "current" : voice,
        group: "voice",
        run: () => sendSettings({ voice }),
      });
    }

    for (const skill of skills) {
      // Skills taking required arguments cannot be run blind from a list; the
      // cortex fills those in when it calls them.
      const runnable = skill.params.length === 0;
      out.push({
        id: `skill-${skill.name}`,
        label: `Run: ${skill.name.replace(/_/g, " ")}`,
        hint: runnable ? skill.description : "needs arguments — ask for it instead",
        group: "skill",
        run: runnable
          ? () => sendSkill(skill.name)
          : () => sendCommand(skill.description.split(".")[0], "text"),
      });
    }

    for (const name of THEMES) {
      out.push({
        id: `theme-${name}`,
        label: `Theme: ${name}`,
        hint: name === theme ? "current" : "colour scheme",
        group: "display",
        run: () => setTheme(name as Theme),
      });
    }

    return out;
  }, [settings, skills, theme, setTheme, setOverlay]);

  const results = useMemo(() => {
    const scored = entries
      .map((entry) => ({
        entry,
        score: Math.max(
          fuzzy(entry.label, query),
          fuzzy(`${entry.group} ${entry.hint}`, query) * 0.4,
        ),
      }))
      .filter((row) => row.score > 0);
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, 40).map((row) => row.entry);
  }, [entries, query]);

  /* open / close */
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "p") {
        event.preventDefault();
        useJarvis.getState().setPaletteOpen(!useJarvis.getState().paletteOpen);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (open) {
      setQuery("");
      setCursor(0);
      // The input mounts with the panel, so focus on the next frame.
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  useEffect(() => {
    setCursor(0);
  }, [query]);

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-index="${cursor}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [cursor]);

  if (!open) return null;

  const choose = (entry: Entry | undefined) => {
    if (!entry) return;
    entry.run();
    setOpen(false);
  };

  return (
    <div className="palette-backdrop" onPointerDown={() => setOpen(false)} role="presentation">
      <div
        className="palette"
        onPointerDown={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
      >
        <input
          ref={inputRef}
          className="palette-input"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="run a command, switch a model, call a skill…"
          spellCheck={false}
          autoComplete="off"
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              setOpen(false);
            } else if (event.key === "ArrowDown") {
              event.preventDefault();
              setCursor((c) => Math.min(results.length - 1, c + 1));
            } else if (event.key === "ArrowUp") {
              event.preventDefault();
              setCursor((c) => Math.max(0, c - 1));
            } else if (event.key === "Enter") {
              event.preventDefault();
              choose(results[cursor]);
            }
          }}
        />
        <div className="palette-list" ref={listRef}>
          {results.map((entry, index) => (
            <button
              key={entry.id}
              type="button"
              data-index={index}
              className={`palette-row ${index === cursor ? "on" : ""}`}
              onPointerEnter={() => setCursor(index)}
              onClick={() => choose(entry)}
            >
              <span className="palette-group">{entry.group}</span>
              <span className="palette-label">{entry.label}</span>
              <span className="palette-hint">{entry.hint}</span>
            </button>
          ))}
          {results.length === 0 && <div className="palette-empty">nothing matches.</div>}
        </div>
        <div className="palette-foot">
          <span>↑↓ navigate</span>
          <span>↵ run</span>
          <span>esc close</span>
        </div>
      </div>
    </div>
  );
}
