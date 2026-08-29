"use client";

/**
 * The reference card.
 *
 * Everything here is reachable by voice, and the command palette indexes the
 * same surface — but neither tells you that Escape is barge-in or that ⌘K
 * arms the wake word, because a keyboard shortcut is the one capability that
 * cannot announce itself. This is the page you open once and then never need
 * again, which is exactly what it should be.
 *
 * The shortcut table is written from the handlers in `Console.tsx`,
 * `CommandPalette.tsx` and `Overlays.tsx`; if a binding changes there, it
 * changes here.
 */

import { useJarvis } from "@/state/jarvis";

/** ⌘ on Apple hardware, Ctrl everywhere else — shown as the user's own key. */
function modifier(): string {
  if (typeof navigator === "undefined") return "Ctrl";
  return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? "⌘" : "Ctrl";
}

const VOICE_GROUPS: Array<{ heading: string; phrases: Array<[string, string]> }> = [
  {
    heading: "CONTROL",
    phrases: [
      ["stop", "cut the assistant off mid-sentence"],
      ["settings", "open the panel"],
      ["status", "spoken system report"],
      ["performance report", "latency and throughput"],
      ["clear the log", "wipe the telemetry panel"],
      ["new conversation", "clear the working context"],
    ],
  },
  {
    heading: "VOICE & MODEL",
    phrases: [
      ["change voice to Bella", "switch the vocal timbre"],
      ["speak faster / slower", "adjust the speaking rate"],
      ["switch model to qwen3", "change the reasoning model"],
      ["list models", "what is installed"],
      ["think harder / answer faster", "extended chain-of-thought on or off"],
    ],
  },
  {
    heading: "DISPOSITION",
    phrases: [
      ["be more concise", "the TERSE persona"],
      ["switch to Friday", "warmer, faster, informal"],
      ["engineering mode", "exact numbers and trade-offs"],
    ],
  },
  {
    heading: "MEMORY",
    phrases: [
      ["remember that I take my coffee black", "store a durable fact"],
      ["what do you know about my coffee", "recall a fact"],
      ["make a note to call the bank", "save a note"],
      ["remind me to stretch in 20 minutes", "schedule a spoken reminder"],
    ],
  },
];

export default function HelpView() {
  const settings = useJarvis((s) => s.settings);
  const skills = useJarvis((s) => s.skills);
  const mod = modifier();

  const shortcuts: Array<[string, string]> = [
    [`${mod} P`, "command palette — the index of everything"],
    [`${mod} K`, "arm the wake word (say “Jarvis, …”)"],
    ["/", "focus the command bar"],
    ["Esc", "barge in: stop speaking, or close a panel"],
    ["↑ ↓", "walk back through directives you have sent"],
    ["Enter", "send the directive"],
  ];

  return (
    <>
      <p className="overlay-note">
        Everything runs on this machine. Nothing leaves it unless you armed a network skill,
        and the audit trail records it when you did.
      </p>

      <div className="overlay-head">KEYBOARD</div>
      <div className="help-keys">
        {shortcuts.map(([key, what]) => (
          <div className="help-key" key={key}>
            <kbd>{key}</kbd>
            <span>{what}</span>
          </div>
        ))}
      </div>

      <div className="overlay-head">SAY ANY OF THIS</div>
      <div className="help-voice">
        {VOICE_GROUPS.map((group) => (
          <div className="help-group" key={group.heading}>
            <div className="help-group-head">{group.heading}</div>
            {group.phrases.map(([phrase, what]) => (
              <div className="help-phrase" key={phrase}>
                <b>“{phrase}”</b>
                <span>{what}</span>
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="overlay-head">RIGHT NOW</div>
      <div className="overlay-kv">
        <span>model</span>
        <b>
          {settings.model || "none"}
          {settings.model_verified ? "" : " (not installed)"}
        </b>
        <span>skills</span>
        <b>
          {skills.length} registered
          {settings.tools_active
            ? " · armed, the cortex calls them itself"
            : settings.tools_supported
              ? " · disabled in settings"
              : " · this model cannot call tools"}
        </b>
        <span>recall</span>
        <b>{settings.recall ? "on — draws on earlier sessions" : "off — this conversation only"}</b>
        <span>thinking</span>
        <b>
          {settings.think_active
            ? "extended — slower, more deliberate"
            : settings.think_supported
              ? "direct — say “think harder” to deliberate"
              : "direct — this model has no thinking mode"}
        </b>
        <span>settings</span>
        <b>
          {settings.persisted?.length
            ? `${settings.persisted.length} saved, restored on next boot`
            : "session only until you change something"}
        </b>
      </div>
    </>
  );
}
