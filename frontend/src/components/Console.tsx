"use client";

/**
 * Bottom console: the live caption of whatever J.A.R.V.I.S. is currently
 * saying (streamed token-by-token while the model generates), plus the
 * command bar - typed input, push-to-talk, and wake-word listening.
 *
 * Keyboard: "/" focuses the input, Esc stops speech (barge-in), and
 * Ctrl/Cmd+K toggles wake-word mode.
 */

import { useEffect, useRef, useState } from "react";
import { audioEngine } from "@/audio/engine";
import { useJarvis } from "@/state/jarvis";
import { sendCommand, sendStop } from "@/hooks/useJarvisConnection";
import { useSpeechRecognition } from "@/hooks/useSpeechRecognition";

const SUGGESTIONS = [
  "status",
  "what time is it",
  "list models",
  "change voice to af heart",
  "run a full system diagnostic",
];

export default function Console() {
  const status = useJarvis((s) => s.status);
  const caption = useJarvis((s) => s.caption);
  const streaming = useJarvis((s) => s.captionStreaming);
  const pushLog = useJarvis((s) => s.pushLog);
  const setStatus = useJarvis((s) => s.setStatus);
  const setMicListening = useJarvis((s) => s.setMicListening);
  const setWakeWord = useJarvis((s) => s.setWakeWord);
  const audioUnlocked = useJarvis((s) => s.audioUnlocked);
  const audioConnected = useJarvis((s) => s.audioConnected);

  const [command, setCommand] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const [historyIdx, setHistoryIdx] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  /** Last dispatched command, to swallow accidental double-sends. */
  const lastSent = useRef<{ text: string; at: number }>({ text: "", at: 0 });

  const dispatch = (text: string, origin: "text" | "voice") => {
    // Two Enter events in the same tick both read the pre-clear `command`
    // state, which would queue the directive twice. Speech recognition can
    // likewise emit a final result twice for one phrase.
    const now = Date.now();
    if (lastSent.current.text === text && now - lastSent.current.at < 1200) return;
    lastSent.current = { text, at: now };

    setStatus("thinking");
    setHistory((h) => [...h.filter((x) => x !== text), text].slice(-40));
    // No optimistic log line here: the backend echoes every accepted command
    // back on /ws/logs (source "you" / "stt") so that every connected client
    // sees it. Logging locally as well printed each directive twice.
    if (!sendCommand(text, origin)) {
      pushLog("error", "client", "uplink unavailable - command dropped");
      setStatus("idle");
    }
  };

  const { listening, wakeMode, interim, supported, toggle, toggleWake } = useSpeechRecognition({
    onTranscript: (text, viaWakeWord) => {
      // Echo guard: on speakers, the mic hears J.A.R.V.I.S. itself and would
      // feed its own answer back in as the next command - an endless loop.
      // Anything captured while audio is still playing is dropped, unless the
      // user deliberately prefixed it with the wake word (which is how you
      // barge in on purpose).
      if (audioEngine.pending > 0.05 && !viaWakeWord) {
        pushLog("info", "stt", "ignored audio captured while speaking (echo guard)");
        return;
      }
      dispatch(text, "voice");
    },
    onError: (message) => pushLog("error", "stt", message),
  });

  useEffect(() => setMicListening(listening), [listening, setMicListening]);
  useEffect(() => setWakeWord(wakeMode), [wakeMode, setWakeWord]);

  /* keyboard shortcuts */
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing = target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName);
      if (event.key === "Escape") {
        sendStop();
        setStatus("idle");
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        toggleWake();
        return;
      }
      if (event.key === "/" && !typing) {
        event.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleWake, setStatus]);

  const submit = () => {
    const text = command.trim();
    if (!text) return;
    setCommand("");
    setHistoryIdx(-1);
    dispatch(text, "text");
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      submit();
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      const next = historyIdx < 0 ? history.length - 1 : Math.max(0, historyIdx - 1);
      if (history[next] !== undefined) {
        setHistoryIdx(next);
        setCommand(history[next]);
      }
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (historyIdx < 0) return;
      const next = historyIdx + 1;
      if (next >= history.length) {
        setHistoryIdx(-1);
        setCommand("");
      } else {
        setHistoryIdx(next);
        setCommand(history[next]);
      }
    }
  };

  const showCaption = caption.length > 0 && (status === "speaking" || streaming);

  return (
    <div className="console">
      {/* live caption of the spoken answer */}
      <div className={`caption ${showCaption ? "on" : ""}`} aria-live="polite">
        <span className="caption-text">
          {caption}
          {streaming && <span className="caret" />}
        </span>
      </div>

      {/* interim speech recognition preview */}
      {interim && (
        <div className="interim">
          <span className="interim-dot" />
          {interim}
        </div>
      )}

      {!audioUnlocked && audioConnected && (
        <button className="audio-unlock" type="button" onClick={() => inputRef.current?.focus()}>
          ⚠ click anywhere to enable audio output
        </button>
      )}

      <div className="cmdbar" data-status={status}>
        <span className="cmd-prompt">❯</span>
        <input
          ref={inputRef}
          value={command}
          onChange={(event) => setCommand(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            status === "boot"
              ? "establishing uplink…"
              : wakeMode
                ? 'listening for "Jarvis…"  ·  or type a directive'
                : "issue a directive   ·   press / to focus"
          }
          spellCheck={false}
          autoComplete="off"
          aria-label="Command input"
        />

        {status === "speaking" && (
          <button className="cmd-btn stop" type="button" onClick={sendStop} title="Stop speaking (Esc)">
            ◼ STOP
          </button>
        )}

        {supported && (
          <>
            <button
              className={`cmd-btn mic ${listening && !wakeMode ? "active" : ""}`}
              onClick={toggle}
              title={listening ? "Stop listening" : "Push to talk"}
              aria-label="Push to talk"
              type="button"
            >
              {listening && !wakeMode ? "◉" : "◎"}
            </button>
            <button
              className={`cmd-btn wake ${wakeMode ? "active" : ""}`}
              onClick={toggleWake}
              title='Wake-word mode - say "Jarvis…" (⌘K)'
              aria-label="Toggle wake word listening"
              type="button"
            >
              {wakeMode ? "◈ WAKE" : "◇ WAKE"}
            </button>
          </>
        )}
      </div>

      {command === "" && status !== "boot" && (
        <div className="suggestions">
          {SUGGESTIONS.map((s) => (
            <button key={s} type="button" className="chip" onClick={() => dispatch(s, "text")}>
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
