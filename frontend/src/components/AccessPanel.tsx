"use client";

/**
 * The three panels you open by hand, gathered under the vitals.
 *
 * They used to be unlabelled glyphs tucked into the telemetry header, sharing
 * a row with the connection state — three of the most-used controls in the
 * interface, rendered as "⌗ ? ⚙" and discoverable only by hovering. Here they
 * get room for their names.
 *
 * Every one of them is reachable another way (`⌘P`, and `?` for the reference
 * card), which is what keeps this safe to hide with the rest of the left rail
 * on a narrow window.
 */

import { useJarvis } from "@/state/jarvis";

const ENTRIES: Array<{ glyph: string; label: string; hint: string; open: () => void }> = [
  {
    glyph: "⌗",
    label: "ARCHIVE",
    hint: "Past conversations & audit trail",
    open: () => useJarvis.getState().setOverlay("archive"),
  },
  {
    glyph: "?",
    label: "REFERENCE",
    hint: "Shortcuts & spoken phrases (?)",
    open: () => useJarvis.getState().setOverlay("help"),
  },
  {
    glyph: "⚙",
    label: "SETTINGS",
    hint: "Model, voice, persona (or say “settings”)",
    open: () => useJarvis.getState().setSettingsOpen(true),
  },
];

export default function AccessPanel() {
  return (
    <aside className="access" aria-label="Panels">
      <div className="access-title">ACCESS</div>
      <div className="access-grid">
        {ENTRIES.map(({ glyph, label, hint, open }) => (
          <button
            key={label}
            type="button"
            className="access-btn"
            onClick={open}
            title={hint}
            aria-label={hint}
          >
            <span className="access-glyph" aria-hidden="true">
              {glyph}
            </span>
            <span className="access-label">{label}</span>
          </button>
        ))}
      </div>
    </aside>
  );
}
