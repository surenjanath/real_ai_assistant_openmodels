"use client";

/**
 * Voice command capture via the browser SpeechRecognition API (progressive
 * enhancement). On Chrome/Safari this gives real STT for the command bar;
 * where unavailable or blocked (e.g. iframe without mic permission) the UI
 * simply hides the mic and typed commands remain fully functional.
 */

import { useCallback, useEffect, useRef, useState } from "react";

type Recognition = {
  start: () => void;
  stop: () => void;
  abort: () => void;
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: any) => void) | null;
  onerror: ((event: any) => void) | null;
  onend: (() => void) | null;
};

function getCtor(): (new () => Recognition) | null {
  if (typeof window === "undefined") return null;
  const w = window as any;
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function useSpeechRecognition(options: {
  onTranscript: (text: string) => void;
  onError?: (message: string) => void;
}) {
  const [listening, setListening] = useState(false);
  const [supported] = useState(() => getCtor() !== null);
  const recognitionRef = useRef<Recognition | null>(null);
  const callbackRef = useRef(options);
  callbackRef.current = options;

  useEffect(() => {
    const Ctor = getCtor();
    if (!Ctor) return;
    const recognition = new Ctor();
    recognition.lang = navigator.language || "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.onresult = (event: any) => {
      const transcript = String(event.results?.[0]?.[0]?.transcript ?? "").trim();
      if (transcript) callbackRef.current.onTranscript(transcript);
    };
    recognition.onerror = (event: any) => {
      const code = String(event?.error ?? "unknown");
      const readable =
        code === "not-allowed" || code === "service-not-allowed"
          ? "microphone permission denied - typed commands remain available"
          : `stt error: ${code}`;
      callbackRef.current.onError?.(readable);
      setListening(false);
    };
    recognition.onend = () => setListening(false);
    recognitionRef.current = recognition;
    return () => {
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      try {
        recognition.abort();
      } catch {
        /* noop */
      }
    };
  }, []);

  const toggle = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    if (listening) {
      recognition.stop();
      setListening(false);
      return;
    }
    try {
      recognition.start();
      setListening(true);
    } catch {
      setListening(false);
    }
  }, [listening]);

  return { listening, supported, toggle };
}
