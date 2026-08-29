"use client";

/**
 * Settings panel - model, voice, persona, skills, memory and output, live.
 *
 * Opened by the ⚙ button in the telemetry header, or by saying "settings"
 * (the backend routes the voice command to a `ui` frame). Changes POST to
 * /api/settings; every connected client then receives a `settings.update`
 * frame, so panels stay in sync and the change is spoken back.
 *
 * Models are split into what Ollama actually has installed versus mere
 * suggestions, because picking an un-pulled tag is the single easiest way to
 * end up with a crew that cannot answer. The same honesty applies to the
 * reasoning toggles: extended thinking and tool calling are both disabled,
 * with the reason shown, when the selected model does not advertise them.
 *
 * Volume lives in the backend registry rather than in local component state,
 * so "speak up" by voice and the slider here move the same number, and every
 * open client agrees on it.
 */

import { useEffect, useState } from "react";
import { audioEngine } from "@/audio/engine";
import { pullModel } from "@/hooks/useJarvisConnection";
import { THEMES, useJarvis, type Theme } from "@/state/jarvis";
import { PROFILES, TIERS, quality, resetQuality, setQuality, type Quality } from "@/state/quality";
import type { LogLevel, PersonaInfo, SettingsState } from "@/lib/protocol";

type SaveState = "idle" | "saving" | "ok" | "error";

/** A disposition being written, before it is saved. `null` = editor closed. */
type Draft = {
  key: string | null; // null while creating; set when editing an existing one
  label: string;
  blurb: string;
  style: string;
  voice: string;
  speed: number;
  temperature: number;
  builtin: boolean;
  edited: boolean;
  custom: boolean;
};

const BLANK: Draft = {
  key: null,
  label: "",
  blurb: "",
  style: "",
  voice: "bm_george",
  speed: 1,
  temperature: 0.6,
  builtin: false,
  edited: false,
  custom: false,
};

function draftOf(p: PersonaInfo): Draft {
  return {
    key: p.key,
    label: p.label,
    blurb: p.blurb,
    style: p.style ?? "",
    voice: p.voice ?? "bm_george",
    speed: p.speed ?? 1,
    temperature: p.temperature ?? 0.6,
    builtin: Boolean(p.builtin),
    edited: Boolean(p.edited),
    custom: Boolean(p.custom),
  };
}

export default function SettingsPanel() {
  const open = useJarvis((s) => s.settingsOpen);
  const setOpen = useJarvis((s) => s.setSettingsOpen);
  const settings = useJarvis((s) => s.settings);
  const setSettings = useJarvis((s) => s.setSettings);
  const engines = useJarvis((s) => s.engines);
  const pushLog = useJarvis((s) => s.pushLog);

  const theme = useJarvis((s) => s.theme);
  const setTheme = useJarvis((s) => s.setTheme);
  const skills = useJarvis((s) => s.skills);
  const memoryStats = useJarvis((s) => s.memoryStats);

  const [save, setSave] = useState<SaveState>("idle");
  const [detail, setDetail] = useState("");
  const [muted, setMuted] = useState(false);
  const pull = useJarvis((s) => s.pull);
  // The tier lives in a mutable singleton so the WebGL tree never re-renders
  // for it; this mirror exists purely to repaint the buttons below.
  const [tier, setTier] = useState<Quality>(quality.tier);
  const [tierAuto, setTierAuto] = useState(quality.auto);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [draftError, setDraftError] = useState("");

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  if (!open) return null;

  const apply = async (delta: {
    model?: string;
    voice?: string;
    speed?: number;
    think?: boolean;
    persona?: string;
    tools?: boolean;
    recall?: boolean;
    volume?: number;
    stream_speech?: boolean;
  }) => {
    setSave("saving");
    setDetail("");
    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(delta),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.errors?.[0] ?? `HTTP ${res.status}`);
      if (data.settings) setSettings(data.settings);
      setSave("ok");
      const changed = Object.entries(data.applied ?? {})
        .map(([k, v]) => `${k} → ${v}`)
        .join(", ");
      setDetail(changed ? `applied ${changed}` : "no change");
      const level: LogLevel = changed ? "success" : "info";
      pushLog(level, "settings", changed ? `panel: ${changed}` : "panel: no changes");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setSave("error");
      setDetail(message);
      pushLog("error", "settings", `panel: ${message}`);
    } finally {
      setTimeout(() => setSave("idle"), 2400);
    }
  };

  /** Create or update a disposition, then adopt it if it is the live one. */
  const saveDraft = async () => {
    if (!draft) return;
    setSave("saving");
    setDraftError("");
    try {
      const res = await fetch("/api/personas", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          key: draft.key ?? undefined,
          label: draft.label,
          blurb: draft.blurb,
          style: draft.style,
          voice: draft.voice,
          speed: draft.speed,
          temperature: draft.temperature,
        }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      setSettings({ personas: data.personas } as Partial<SettingsState>);
      pushLog("success", "settings", `disposition '${draft.label}' saved`);
      setDraft(null);
      setSave("ok");
      setDetail(`saved ${draft.label}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setDraftError(message);
      setSave("error");
    } finally {
      setTimeout(() => setSave("idle"), 2400);
    }
  };

  /** Delete a custom disposition, or reset an edited built-in to how it shipped. */
  const removeDraft = async () => {
    if (!draft?.key) return;
    setSave("saving");
    setDraftError("");
    try {
      const res = await fetch(`/api/personas/${encodeURIComponent(draft.key)}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      setSettings({ personas: data.personas } as Partial<SettingsState>);
      pushLog("info", "settings", `disposition '${draft.key}' ${data.result}`);
      setDraft(null);
      setSave("ok");
      setDetail(`${draft.key} ${data.result}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setDraftError(message);
      setSave("error");
    } finally {
      setTimeout(() => setSave("idle"), 2400);
    }
  };

  const testVoice = () => {
    audioEngine.unlock();
    void fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: "Voice check. All systems are functioning within normal parameters.",
      }),
    }).catch(() => undefined);
  };

  const statusText =
    save === "saving" ? "APPLYING…" : save === "ok" ? "APPLIED ✓" : save === "error" ? "REJECTED" : "";

  const installed = settings.installed ?? [];
  const suggestions = (settings.models ?? []).filter((m) => !installed.includes(m));
  const label = (voice: string) => settings.voice_labels?.[voice] ?? voice;

  return (
    <>
      <div className="settings-scrim" onClick={() => setOpen(false)} />
      <section className="settings-panel" role="dialog" aria-label="Settings">
        <header className="telemetry-header">
          <span className="telemetry-title">SYSTEM CONFIGURATION</span>
          <button
            className="settings-close"
            onClick={() => setOpen(false)}
            aria-label="Close settings"
            type="button"
          >
            ✕
          </button>
        </header>

        <div className="settings-body">
          <label className="settings-field">
            <span className="settings-label">
              CREW MODEL
              <em className={`settings-note ${settings.model_verified ? "" : "bad"}`}>
                {settings.model_verified
                  ? `installed on ollama · ${installed.length} available`
                  : "NOT installed — pull it or pick one below"}
              </em>
            </span>
            <select
              value={settings.model}
              onChange={(event) => apply({ model: event.target.value })}
              disabled={save === "saving"}
            >
              {!settings.models.includes(settings.model) && (
                <option value={settings.model}>{settings.model}</option>
              )}
              {installed.length > 0 && (
                <optgroup label="Installed">
                  {installed.map((model) => (
                    <option key={model} value={model}>
                      {model}
                      {model === settings.model ? "  ●" : ""}
                    </option>
                  ))}
                </optgroup>
              )}
              {suggestions.length > 0 && (
                <optgroup label="Not installed (ollama pull …)">
                  {suggestions.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            {!settings.model_verified && settings.model && (
              <button
                type="button"
                className="settings-action"
                disabled={Boolean(pull)}
                onClick={() => {
                  void pullModel(settings.model);
                  pushLog("info", "models", `requested download of ${settings.model}`);
                }}
              >
                {pull
                  ? `pulling ${pull.model}${typeof pull.percent === "number" ? ` · ${pull.percent}%` : "…"}`
                  : `↓ install ${settings.model}`}
              </button>
            )}
          </label>

          <label className="settings-field">
            <span className="settings-label">
              VOICE
              <em className="settings-note">
                {engines.ttsMode === "kokoro"
                  ? "kokoro-82M · applies on the next utterance"
                  : "fallback synth — install pykokoro for real speech"}
              </em>
            </span>
            <select
              value={settings.voice}
              onChange={(event) => apply({ voice: event.target.value })}
              disabled={save === "saving"}
            >
              {!settings.voices.includes(settings.voice) && (
                <option value={settings.voice}>{settings.voice}</option>
              )}
              {settings.voices.map((voice) => (
                <option key={voice} value={voice}>
                  {label(voice)}
                  {voice === settings.voice ? "  ●" : ""}
                </option>
              ))}
            </select>
          </label>

          <label className="settings-field">
            <span className="settings-label">
              SPEECH SPEED <b className="settings-value">{settings.speed.toFixed(2)}×</b>
            </span>
            <input
              type="range"
              min={0.5}
              max={2}
              step={0.05}
              value={settings.speed}
              onChange={(event) => setSettings({ speed: Number(event.target.value) })}
              onPointerUp={(event) => apply({ speed: Number((event.target as HTMLInputElement).value) })}
              onKeyUp={(event) => apply({ speed: Number((event.target as HTMLInputElement).value) })}
              disabled={save === "saving"}
            />
            <span className="settings-scale">
              <span>0.5×</span>
              <span>1.0×</span>
              <span>2.0×</span>
            </span>
          </label>

          <label className="settings-field">
            <span className="settings-label">
              OUTPUT VOLUME <b className="settings-value">{Math.round(settings.volume * 100)}%</b>
              <em className="settings-note">shared with “louder” / “quieter” by voice</em>
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={settings.volume}
              onChange={(event) => {
                // Apply locally on every drag frame so it is audible at once,
                // and persist to the registry only when the gesture ends.
                const v = Number(event.target.value);
                setSettings({ volume: v });
                audioEngine.setVolume(v);
              }}
              onPointerUp={(event) => apply({ volume: Number((event.target as HTMLInputElement).value) })}
              onKeyUp={(event) => apply({ volume: Number((event.target as HTMLInputElement).value) })}
            />
          </label>

          <div className="settings-field">
            <span className="settings-label">
              COLOUR SCHEME
              <em className="settings-note">interface only — remembered on this device</em>
            </span>
            <div className="theme-row">
              {THEMES.map((name) => (
                <button
                  key={name}
                  type="button"
                  className={`theme-chip ${theme === name ? "on" : ""}`}
                  data-theme-swatch={name}
                  onClick={() => setTheme(name as Theme)}
                >
                  {name}
                </button>
              ))}
            </div>
          </div>

          <div className="settings-field">
            <span className="settings-label">
              DISPOSITION
              <em className="settings-note">
                the same crew, a different manner — switches instantly
              </em>
            </span>
            <div className="persona-grid">
              {(settings.personas ?? []).map((persona) => (
                <div
                  key={persona.key}
                  className={`persona ${settings.persona === persona.key ? "on" : ""}`}
                >
                  {/* Selecting and editing are separate targets. A single card
                      that did both would make every edit a disposition switch
                      as a side effect. */}
                  <button
                    type="button"
                    className="persona-pick"
                    onClick={() => apply({ persona: persona.key })}
                    disabled={save === "saving"}
                    title={persona.blurb}
                  >
                    <b>{persona.label}</b>
                    <em>{persona.blurb}</em>
                    {/* Choosing a disposition moves the voice with it, so the
                        card says which one rather than letting the VOICE control
                        appear to change on its own. */}
                    {persona.voice && (
                      <i className="persona-voice">
                        {settings.voice_labels?.[persona.voice] ?? persona.voice}
                        {persona.custom ? " · yours" : persona.edited ? " · edited" : ""}
                      </i>
                    )}
                  </button>
                  <button
                    type="button"
                    className="persona-edit"
                    onClick={() => {
                      setDraftError("");
                      setDraft(draftOf(persona));
                    }}
                    title={`Edit ${persona.label}`}
                    aria-label={`Edit ${persona.label}`}
                  >
                    ✎
                  </button>
                </div>
              ))}
              <button
                type="button"
                className="persona persona-new"
                onClick={() => {
                  setDraftError("");
                  setDraft({ ...BLANK, voice: settings.voice ?? BLANK.voice });
                }}
                title="Write a disposition of your own"
              >
                <b>+ NEW</b>
                <em>Write your own manner</em>
              </button>
            </div>

            {draft && (
              <div className="persona-editor">
                <div className="persona-editor-head">
                  <b>{draft.key ? `EDIT ${draft.label.toUpperCase()}` : "NEW DISPOSITION"}</b>
                  {draft.builtin && (
                    <em>
                      built in — your changes are saved alongside it and can be undone
                    </em>
                  )}
                </div>

                <label className="persona-row">
                  <span>NAME</span>
                  <input
                    value={draft.label}
                    maxLength={24}
                    placeholder="PIRATE"
                    onChange={(e) => setDraft({ ...draft, label: e.target.value })}
                  />
                </label>

                <label className="persona-row">
                  <span>SUMMARY</span>
                  <input
                    value={draft.blurb}
                    maxLength={120}
                    placeholder="What it is like, in a few words"
                    onChange={(e) => setDraft({ ...draft, blurb: e.target.value })}
                  />
                </label>

                <label className="persona-row tall">
                  <span>MANNER</span>
                  <textarea
                    value={draft.style}
                    maxLength={2000}
                    rows={6}
                    placeholder="Written to the model, in the second person: “Be warm and quick. Two sentences.” Examples of the finished register work far better than rules."
                    onChange={(e) => setDraft({ ...draft, style: e.target.value })}
                  />
                </label>

                <label className="persona-row">
                  <span>VOICE</span>
                  <select
                    value={draft.voice}
                    onChange={(e) => setDraft({ ...draft, voice: e.target.value })}
                  >
                    {(settings.voices ?? []).map((v) => (
                      <option key={v} value={v}>
                        {settings.voice_labels?.[v] ?? v} · {v}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="persona-row">
                  <span>RATE {draft.speed.toFixed(2)}×</span>
                  <input
                    type="range"
                    min={0.5}
                    max={2}
                    step={0.01}
                    value={draft.speed}
                    onChange={(e) => setDraft({ ...draft, speed: Number(e.target.value) })}
                  />
                </label>

                <label className="persona-row">
                  <span>WARMTH {draft.temperature.toFixed(2)}</span>
                  <input
                    type="range"
                    min={0}
                    max={1.5}
                    step={0.05}
                    value={draft.temperature}
                    onChange={(e) =>
                      setDraft({ ...draft, temperature: Number(e.target.value) })
                    }
                  />
                </label>

                {draftError && <div className="persona-error">{draftError}</div>}

                <div className="persona-actions">
                  <button
                    type="button"
                    className="settings-btn on"
                    onClick={saveDraft}
                    disabled={save === "saving"}
                  >
                    SAVE
                  </button>
                  <button
                    type="button"
                    className="settings-btn"
                    onClick={() => {
                      setDraft(null);
                      setDraftError("");
                    }}
                  >
                    CANCEL
                  </button>
                  {/* A built-in is reset, never removed: a saved preference or
                      a spoken phrase may still name it. */}
                  {(draft.custom || draft.edited) && (
                    <button
                      type="button"
                      className="settings-btn danger"
                      onClick={removeDraft}
                      disabled={save === "saving"}
                    >
                      {draft.edited ? "RESET TO DEFAULT" : "DELETE"}
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>

          <label className="settings-field">
            <span className="settings-label">
              SKILLS
              <em className={`settings-note ${settings.tools_supported ? "" : "bad"}`}>
                {settings.tools_supported
                  ? `${skills.length} available — the cortex calls them itself`
                  : "this model does not advertise tool calling"}
              </em>
            </span>
            <div className="toggle-row">
              <button
                type="button"
                className={`toggle ${!settings.tools ? "on" : ""}`}
                onClick={() => apply({ tools: false })}
                disabled={save === "saving"}
              >
                OFF · model only
              </button>
              <button
                type="button"
                className={`toggle ${settings.tools ? "on" : ""}`}
                onClick={() => apply({ tools: true })}
                disabled={save === "saving" || !settings.tools_supported}
              >
                ON · armed
              </button>
            </div>
          </label>

          <label className="settings-field">
            <span className="settings-label">
              SPEECH START
              <em className="settings-note">
                {settings.stream_speech
                  ? "speaks each sentence as it is written"
                  : "waits for the whole answer first"}
              </em>
            </span>
            <div className="toggle-row">
              <button
                type="button"
                className={`toggle ${!settings.stream_speech ? "on" : ""}`}
                onClick={() => apply({ stream_speech: false })}
                disabled={save === "saving"}
              >
                OFF · whole answer
              </button>
              <button
                type="button"
                className={`toggle ${settings.stream_speech ? "on" : ""}`}
                onClick={() => apply({ stream_speech: true })}
                disabled={save === "saving"}
              >
                ON · per sentence
              </button>
            </div>
          </label>

          <label className="settings-field">
            <span className="settings-label">
              RENDER QUALITY
              <em className="settings-note">
                {PROFILES[tier].particles.toLocaleString()} particles
                {tierAuto ? " · chosen automatically" : " · set by you"}
                {quality.fps ? ` · ${quality.fps} fps` : ""}
              </em>
            </span>
            <div className="toggle-row">
              {TIERS.map((option) => (
                <button
                  key={option}
                  type="button"
                  className={`toggle ${tier === option ? "on" : ""}`}
                  onClick={() => {
                    setQuality(option);
                    setTier(option);
                    setTierAuto(false);
                  }}
                >
                  {PROFILES[option].label}
                </button>
              ))}
              <button
                type="button"
                className={`toggle ${tierAuto ? "on" : ""}`}
                title="Pick from the hardware and correct it from the measured frame rate"
                onClick={() => {
                  resetQuality();
                  setTier(quality.tier);
                  setTierAuto(true);
                }}
              >
                auto
              </button>
            </div>
          </label>

          <label className="settings-field">
            <span className="settings-label">
              LONG-TERM RECALL
              <em className="settings-note">
                {memoryStats
                  ? `${memoryStats.turns} exchanges across ${memoryStats.sessions} sessions`
                  : "durable memory across restarts"}
              </em>
            </span>
            <div className="toggle-row">
              <button
                type="button"
                className={`toggle ${!settings.recall ? "on" : ""}`}
                onClick={() => apply({ recall: false })}
                disabled={save === "saving"}
              >
                OFF · this session
              </button>
              <button
                type="button"
                className={`toggle ${settings.recall ? "on" : ""}`}
                onClick={() => apply({ recall: true })}
                disabled={save === "saving"}
              >
                ON · remembers
              </button>
            </div>
          </label>

          <label className="settings-field">
            <span className="settings-label">
              EXTENDED THINKING
              <em className={`settings-note ${settings.think_supported ? "" : "bad"}`}>
                {settings.think_supported
                  ? "slower, more careful answers"
                  : "this model has no thinking mode"}
              </em>
            </span>
            <div className="toggle-row">
              <button
                type="button"
                className={`toggle ${!settings.think ? "on" : ""}`}
                onClick={() => apply({ think: false })}
                disabled={save === "saving"}
              >
                OFF · fast
              </button>
              <button
                type="button"
                className={`toggle ${settings.think ? "on" : ""}`}
                onClick={() => apply({ think: true })}
                disabled={save === "saving" || !settings.think_supported}
                title={
                  settings.think_supported
                    ? "Let the model reason before answering"
                    : "Unavailable on this model"
                }
              >
                ON · deep
              </button>
            </div>
          </label>

          <div className="settings-actions">
            <button type="button" className="settings-btn" onClick={testVoice}>
              ▶ Test voice
            </button>
            <button
              type="button"
              className={`settings-btn ${muted ? "on" : ""}`}
              onClick={() => {
                const next = !muted;
                setMuted(next);
                audioEngine.setMuted(next);
              }}
            >
              {muted ? "🔇 Muted" : "🔊 Sound on"}
            </button>
          </div>

          <div className="settings-status">
            <span>tts/{engines.ttsLabel || engines.tts}</span>
            <span>crew/{engines.agents}</span>
            {engines.mode && <span>mode/{engines.mode}</span>}
          </div>

          <p className="settings-hint">
            Voice: <i>“settings”</i>, <i>“switch model to qwen3 8b”</i>, <i>“change voice to
            af heart”</i>, <i>“be more concise”</i>, <i>“remember that …”</i>, <i>“what do
            you know about …”</i>, <i>“remind me to … in ten minutes”</i>,{" "}
            <i>“performance report”</i>, <i>“stop”</i>.
            <br />
            Keys: <b>/</b> focus · <b>⌘P</b> command palette · <b>⌘K</b> wake word ·{" "}
            <b>Esc</b> stop speech.
          </p>
        </div>

        <footer className="settings-footer">
          <span className={`settings-save ${save === "error" ? "bad" : "good"}`}>{statusText}</span>
          <span className="settings-detail">{detail}</span>
        </footer>
      </section>
    </>
  );
}
