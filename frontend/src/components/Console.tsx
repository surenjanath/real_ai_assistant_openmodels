"use client";

/**
 * Bottom console: the live caption of whatever J.A.R.V.I.S. is currently
 * saying (streamed token-by-token while the model generates), plus the
 * command bar - typed input, push-to-talk, and wake-word listening.
 *
 * Keyboard: "/" focuses the input, Esc stops speech (barge-in), Ctrl/Cmd+K
 * toggles wake-word mode, and "?" opens the reference card.
 */

import { useEffect, useRef, useState } from "react";
import { audioEngine } from "@/audio/engine";
import { micMonitor } from "@/audio/mic";
import { looksLikeEcho } from "@/lib/echo";
import { useJarvis } from "@/state/jarvis";
import { sendCommand, sendStop } from "@/hooks/useJarvisConnection";
import { useSpeechRecognition } from "@/hooks/useSpeechRecognition";

/**
 * How long after the last sample the speakers are still treated as "loud".
 *
 * Covers the tail of the room's reverberation plus the recogniser's own input
 * buffering, both of which outlive the audio itself.
 */
const OUTPUT_HANGOVER_MS = 450;

/** Deliberately chosen to demonstrate the capabilities that are not obvious:
 *  durable memory, the reflex arc, skills, and the disposition switch. */
const SUGGESTIONS = [
  "status",
  "remember that I take my coffee black",
  "how much memory is this machine using right now",
  "remind me to stretch in 20 minutes",
  "how many kilometres is a marathon",
  "be more concise",
  "performance report",
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

  const dispatch = (text: string, origin: "text" | "voice", verified = false) => {
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
    if (!sendCommand(text, origin, verified)) {
      pushLog("error", "client", "uplink unavailable - command dropped");
      setStatus("idle");
    }
  };

  const { listening, wakeMode, interim, interimStartedAt, supported, toggle, toggleWake } =
    useSpeechRecognition({
      onTranscript: ({ text, startedAt }) => {
        // Echo guard. On speakers the microphone hears J.A.R.V.I.S. as well as
        // the operator, and without this the assistant answers itself, forever.
        //
        // Everything here turns on `startedAt` - when the words were
        // *captured*, not when the transcript arrived. Recognition finalises a
        // phrase a second or more after the sound, by which point the
        // utterance it echoed has often stopped playing, and an "is audio
        // playing right now" check waves every echo straight through.
        const overSpeech = audioEngine.audibleWithin(startedAt, OUTPUT_HANGOVER_MS);
        // The echo-cancelled capture stream is the only thing that can tell
        // the operator from the speakers: it has the assistant's own output
        // subtracted out, so energy left in it means a person in the room.
        const human = micMonitor.running && micMonitor.voiceSince(startedAt);

        if (overSpeech) {
          // Captured while the assistant was talking. Take it only if a
          // person demonstrably spoke, and even then not if it is made of the
          // words we were saying at the time - which is what leaked echo
          // looks like when cancellation is imperfect.
          if (!human || looksLikeEcho(text)) {
            pushLog("info", "stt", `echo suppressed - "${text.slice(0, 48)}"`);
            return;
          }
          // A genuine interruption. Cut the answer off locally before the
          // round trip, so the room goes quiet immediately rather than a
          // network hop later.
          pushLog("voice", "stt", "barge-in - stopping playback");
          sendStop();
        } else if (!human && looksLikeEcho(text)) {
          // Nothing was playing as far as we can tell, and nobody was heard
          // to speak - yet here is a transcript made of what we just said.
          // Headphones unplugged mid-answer, echo cancellation unavailable, a
          // second machine in the room: the acoustics lied, the content did
          // not. Note this only fires when the microphone heard *no one*; a
          // person repeating the assistant's own suggestion back to it
          // ("open the settings panel") is a real directive and gets through.
          pushLog("info", "stt", `echo suppressed on content - "${text.slice(0, 48)}"`);
          return;
        }

        // `human` travels with the directive: the backend can only judge an
        // echo by content, and content alone cannot tell the operator
        // repeating a suggestion back from the assistant being overheard.
        dispatch(text, "voice", human);
      },
      onError: (message) => pushLog("error", "stt", message),
    });

  /* Echo-cancelled capture, opened alongside the recogniser's own stream and
   * used only as a loudness meter. It is what makes barge-in possible: without
   * it, anything heard while the assistant talks has to be discarded, because
   * there is no way to tell the operator's voice from the speakers'. */
  const micWarned = useRef(false);
  useEffect(() => {
    if (!listening && !wakeMode) return;
    let live = true;
    void micMonitor
      .start({
        // React the instant the room gets loud, rather than a second later
        // when the recogniser has made up its mind: drop the answer to a
        // murmur so the operator is not talked over while they interrupt.
        // Ducking, not stopping - a cough should not cost you the answer.
        onVoiceStart: () => {
          if (audioEngine.audibleWithin(Date.now(), 0)) audioEngine.setDucked(true);
        },
        onVoiceEnd: () => audioEngine.setDucked(false),
      })
      .then((ok) => {
        if (!live || ok || micWarned.current) return;
        micWarned.current = true;
        pushLog(
          "warn",
          "stt",
          "no echo-cancelled capture - interrupting by voice is unavailable, " +
            "speech heard while J.A.R.V.I.S. talks will be discarded",
        );
      });
    return () => {
      live = false;
    };
  }, [listening, wakeMode, pushLog]);

  /* Wake-word mode is the only one that leaves the microphone open unattended;
   * when it is switched off, release the capture stream with it. */
  useEffect(() => {
    if (wakeMode || listening) return;
    const timer = setTimeout(() => {
      if (!useJarvis.getState().wakeWordOn && !useJarvis.getState().micListening) {
        micMonitor.stop();
        audioEngine.setDucked(false);
      }
    }, 1500);
    return () => clearTimeout(timer);
  }, [wakeMode, listening]);

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
      // "?" is the conventional key for "what can I press here" — but only
      // when it is not simply a character being typed into the command bar.
      if (event.key === "?" && !typing && !event.metaKey && !event.ctrlKey) {
        event.preventDefault();
        useJarvis.getState().setOverlay("help");
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

      {/* interim speech recognition preview - hidden while the phrase behind
          it was captured over the assistant's own voice, so an echo never
          flashes up on screen as though the operator had said it */}
      {interim && !audioEngine.audibleWithin(interimStartedAt, OUTPUT_HANGOVER_MS) && (
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
                : "issue a directive   ·   press / to focus   ·   ⌘P for commands"
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
