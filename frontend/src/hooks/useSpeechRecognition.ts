"use client";

/**
 * Voice capture via the browser SpeechRecognition API (progressive
 * enhancement). On Chrome this gives real STT; where unavailable or blocked
 * the UI hides the mic and typed commands remain fully functional.
 *
 * Two modes:
 *   - push-to-talk  : one utterance per click.
 *   - wake word     : continuous listening; a transcript is only dispatched
 *                     when it starts with "jarvis" (or the assistant is
 *                     already mid-conversation with you). This is what makes
 *                     it feel like the real thing - you talk to the room.
 *
 * Chrome ends a continuous session on its own every so often (and on silence),
 * so wake-word mode restarts the recogniser automatically until switched off.
 */

import { useCallback, useEffect, useRef, useState } from "react";

type Recognition = {
  start: () => void;
  stop: () => void;
  abort: () => void;
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((event: any) => void) | null;
  onerror: ((event: any) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
};

function getCtor(): (new () => Recognition) | null {
  if (typeof window === "undefined") return null;
  const w = window as any;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

const WAKE = /^(?:hey\s+|ok(?:ay)?\s+)?jarvis\b[\s,.:!-]*/i;

export interface SpeechOptions {
  onTranscript: (text: string, viaWakeWord: boolean) => void;
  onInterim?: (text: string) => void;
  onError?: (message: string) => void;
  /** Speak-anything mode: dispatch even without the wake word. */
  alwaysDispatch?: boolean;
}

export function useSpeechRecognition(options: SpeechOptions) {
  const [listening, setListening] = useState(false);
  const [wakeMode, setWakeMode] = useState(false);
  const [interim, setInterim] = useState("");
  // Must start false so the server-rendered markup matches the first client
  // render; SpeechRecognition only exists in the browser, and deciding this
  // during render would tear down hydration (and with it the WebGL canvas).
  const [supported, setSupported] = useState(false);
  useEffect(() => setSupported(getCtor() !== null), []);

  const recognitionRef = useRef<Recognition | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;
  /** Mirrors `wakeMode` for callbacks that must not be re-bound each render. */
  const wakeRef = useRef(false);
  const stoppingRef = useRef(false);
  const restartTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    const Ctor = getCtor();
    if (!Ctor) return;
    const recognition = new Ctor();
    recognition.lang = navigator.language || "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setListening(true);

    recognition.onresult = (event: any) => {
      let finalText = "";
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const text = String(result[0]?.transcript ?? "");
        if (result.isFinal) finalText += text;
        else interimText += text;
      }

      if (interimText) {
        setInterim(interimText.trim());
        optionsRef.current.onInterim?.(interimText.trim());
      }

      const text = finalText.trim();
      if (!text) return;
      setInterim("");

      if (wakeRef.current && !optionsRef.current.alwaysDispatch) {
        const match = WAKE.exec(text);
        if (!match) return; // ambient speech - ignore it entirely
        const command = text.slice(match[0].length).trim();
        if (command) optionsRef.current.onTranscript(command, true);
        return;
      }
      // Push-to-talk: strip a wake word if the user said one out of habit.
      optionsRef.current.onTranscript(text.replace(WAKE, "").trim() || text, false);
    };

    recognition.onerror = (event: any) => {
      const code = String(event?.error ?? "unknown");
      if (code === "no-speech" || code === "aborted") return; // routine
      const readable =
        code === "not-allowed" || code === "service-not-allowed"
          ? "microphone permission denied - typed commands remain available"
          : `stt error: ${code}`;
      optionsRef.current.onError?.(readable);
      if (code === "not-allowed" || code === "service-not-allowed") {
        wakeRef.current = false;
        setWakeMode(false);
      }
    };

    recognition.onend = () => {
      setListening(false);
      setInterim("");
      // Chrome ends continuous sessions periodically; revive wake-word mode.
      if (wakeRef.current && !stoppingRef.current) {
        restartTimer.current = setTimeout(() => {
          try {
            recognition.start();
          } catch {
            /* already starting */
          }
        }, 320);
      }
    };

    recognitionRef.current = recognition;
    return () => {
      stoppingRef.current = true;
      clearTimeout(restartTimer.current);
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      recognition.onstart = null;
      try {
        recognition.abort();
      } catch {
        /* noop */
      }
    };
  }, []);

  const start = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    stoppingRef.current = false;
    try {
      recognition.start();
    } catch {
      /* already running */
    }
  }, []);

  const stop = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    stoppingRef.current = true;
    clearTimeout(restartTimer.current);
    try {
      recognition.stop();
    } catch {
      /* noop */
    }
    setListening(false);
  }, []);

  /** Push-to-talk toggle. */
  const toggle = useCallback(() => {
    if (listening) stop();
    else start();
  }, [listening, start, stop]);

  /** Continuous wake-word listening toggle. */
  const toggleWake = useCallback(() => {
    const next = !wakeRef.current;
    wakeRef.current = next;
    setWakeMode(next);
    if (next) start();
    else stop();
  }, [start, stop]);

  return { listening, wakeMode, interim, supported, toggle, toggleWake, start, stop };
}
