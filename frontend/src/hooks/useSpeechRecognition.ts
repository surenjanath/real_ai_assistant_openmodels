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
 *
 * ## Endpointing
 *
 * Chrome decides a phrase is over on its own schedule, and it is a slow one:
 * `isFinal` typically lands a second or more after the operator stopped
 * talking, and that second is dead air at the front of every single spoken
 * exchange - before the model has been asked anything at all.
 *
 * So we do our own endpointing. Interim results arrive continuously; once one
 * has stopped changing for long enough that the phrase is plainly over, we
 * dispatch it immediately rather than waiting for the engine to agree. How
 * long "long enough" is depends on the phrase (see `endpointDelay`) - the cost
 * of guessing wrong is cutting the operator off mid-sentence, so short and
 * obviously-unfinished phrases are given more room. Even the most impatient
 * of those holds is well under what Chrome would have taken.
 *
 * Every phrase is stamped with the moment its first audio was heard. That
 * timestamp is the only sound basis for echo suppression: what matters is
 * whether the assistant was talking when the words were *captured*, not when
 * the transcript finally arrived.
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

/** Words that almost never end a sentence - hearing one means keep waiting. */
const DANGLING = new Set(
  `a an the and or but so to of in on at for with from by is are was were am be
   been being do does did can could will would should may might must my your his
   her its our their this that these those what which who how when where why if
   about into over under than then as it i you we they me him them`.split(/\s+/),
);

/** Silence after an obviously unfinished phrase, in ms. */
const HOLD_DANGLING = 1300;
/** Silence after a one- or two-word phrase, in ms. */
const HOLD_SHORT = 950;
/** Silence after a short but plausibly complete phrase, in ms. */
const HOLD_BRIEF = 750;
/** Silence after a phrase long enough to be a whole thought, in ms. */
const HOLD_NORMAL = 600;
/** Gap before a dropped continuous session is revived, in ms. */
const RESTART_MS = 120;

function wordsOf(text: string): string[] {
  return text.trim().toLowerCase().match(/[a-z0-9']+/g) ?? [];
}

/**
 * How much silence must follow this text before we call the phrase finished.
 *
 * Longer phrases are safe to cut early: the operator has already said enough
 * to act on, and a stray extra word is far less costly than a second of dead
 * air on every exchange. Short or dangling ones are almost certainly still in
 * progress, so they are given room to finish.
 */
function endpointDelay(text: string): number {
  const words = wordsOf(text);
  if (words.length === 0) return Number.POSITIVE_INFINITY;
  if (DANGLING.has(words[words.length - 1])) return HOLD_DANGLING;
  if (words.length <= 2) return HOLD_SHORT;
  if (words.length <= 5) return HOLD_BRIEF;
  return HOLD_NORMAL;
}

/** One captured phrase, with the timing the echo guard needs. */
export interface Phrase {
  text: string;
  /** the transcript began with the wake word */
  viaWakeWord: boolean;
  /** wall-clock ms at which the first audio of this phrase was heard */
  startedAt: number;
  /** wall-clock ms at which the phrase was judged complete */
  endedAt: number;
  /** true when our own endpointer called it, rather than the engine */
  early: boolean;
}

export interface SpeechOptions {
  onTranscript: (phrase: Phrase) => void;
  onInterim?: (text: string) => void;
  onError?: (message: string) => void;
  /** Speak-anything mode: dispatch even without the wake word. */
  alwaysDispatch?: boolean;
}

export function useSpeechRecognition(options: SpeechOptions) {
  const [listening, setListening] = useState(false);
  const [wakeMode, setWakeMode] = useState(false);
  const [interim, setInterim] = useState("");
  /** When the audio behind the current interim text started arriving. */
  const [interimStartedAt, setInterimStartedAt] = useState(0);
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

  /* ---- per-phrase state, all in refs: none of it belongs to a render ---- */

  /** Results already folded into a final transcript. */
  const consumed = useRef(0);
  /** Wall-clock ms of the first audio of the phrase in progress, 0 if none. */
  const phraseStart = useRef(0);
  /** Latest interim text for the phrase in progress. */
  const phraseText = useRef("");
  /** Text our endpointer already dispatched for this phrase, normalised. */
  const emitted = useRef("");
  /** The wake word has been heard, so the rest of this phrase is addressed
   *  to us even though it will not repeat the name. */
  const wakeArmed = useRef(false);
  const endpointTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    const Ctor = getCtor();
    if (!Ctor) return;
    const recognition = new Ctor();
    recognition.lang = navigator.language || "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    const clearPhrase = () => {
      clearTimeout(endpointTimer.current);
      endpointTimer.current = undefined;
      phraseStart.current = 0;
      phraseText.current = "";
      emitted.current = "";
      wakeArmed.current = false;
      setInterim("");
      setInterimStartedAt(0);
    };

    /** Apply wake-word policy and hand the phrase on, or drop it. */
    const dispatch = (raw: string, early: boolean) => {
      const text = raw.trim();
      if (!text) return;
      const startedAt = phraseStart.current || Date.now();
      const emit = (body: string, viaWakeWord: boolean) =>
        optionsRef.current.onTranscript({
          text: body,
          viaWakeWord,
          startedAt,
          endedAt: Date.now(),
          early,
        });

      if (wakeRef.current && !optionsRef.current.alwaysDispatch) {
        const match = WAKE.exec(text);
        if (match) {
          // Heard the name. Anything still to come in this phrase is meant
          // for us too - "Jarvis…" followed by a pause and then the actual
          // directive is how people naturally speak, and our own endpointer
          // will have closed the phrase on that pause. Without this the name
          // dispatches an empty command and the directive behind it is thrown
          // away as ambient room noise.
          wakeArmed.current = true;
          const command = text.slice(match[0].length).trim();
          if (command) emit(command, true);
          return;
        }
        if (!wakeArmed.current) return; // ambient speech - ignore it entirely
        emit(text, true);
        return;
      }
      // Push-to-talk: strip a wake word if the user said one out of habit.
      emit(text.replace(WAKE, "").trim() || text, false);
    };

    /** Our own endpointer fired: the phrase has stopped changing. */
    const endpoint = () => {
      endpointTimer.current = undefined;
      const text = phraseText.current.trim();
      if (!text) return;
      emitted.current = text.toLowerCase();
      setInterim("");
      dispatch(text, true);
    };

    recognition.onstart = () => setListening(true);

    recognition.onresult = (event: any) => {
      let finalText = "";
      let interimText = "";
      // Walk from the first result we have not already turned into a final
      // transcript, rather than from `resultIndex`: interim text for one
      // phrase can span several result slots, and only tracking the changed
      // one loses the front of a long sentence.
      if (consumed.current > event.results.length) consumed.current = 0;
      for (let i = consumed.current; i < event.results.length; i++) {
        const result = event.results[i];
        const text = String(result[0]?.transcript ?? "");
        if (result.isFinal) {
          finalText += text;
          consumed.current = i + 1;
        } else {
          interimText += text;
        }
      }

      const now = Date.now();
      if (!phraseStart.current && (interimText.trim() || finalText.trim())) {
        // First sign of this phrase. Interim results lag the sound that
        // produced them, so bias the stamp backwards - the echo guard needs
        // to know when the microphone heard it, not when Chrome said so.
        phraseStart.current = now - 250;
        setInterimStartedAt(phraseStart.current);
      }

      if (interimText.trim()) {
        const text = interimText.trim();
        phraseText.current = text;
        setInterim(text);
        optionsRef.current.onInterim?.(text);
        // Re-arm: every revision means the operator is still talking.
        clearTimeout(endpointTimer.current);
        const delay = endpointDelay(text);
        if (Number.isFinite(delay)) {
          endpointTimer.current = setTimeout(endpoint, delay);
        }
      }

      const text = finalText.trim();
      if (!text) return;
      clearTimeout(endpointTimer.current);
      endpointTimer.current = undefined;

      const already = emitted.current;
      if (!already) {
        // The engine beat our endpointer to it - the fast path when the
        // operator's phrase ends decisively.
        setInterim("");
        dispatch(text, false);
        clearPhrase();
        return;
      }

      // We already dispatched this phrase early. Anything the engine adds on
      // top is either a rounding difference (drop it - re-dispatching would
      // run the directive twice) or genuinely more speech that arrived after
      // we committed, in which case it is a directive of its own.
      const lower = text.toLowerCase();
      const tail = lower.startsWith(already) ? text.slice(already.length).trim() : "";
      const armed = wakeArmed.current;
      clearPhrase();
      // Two words is a plausible completion of what we already said ("…in
      // london"); three is a directive of its own that arrived after we
      // committed. Only the latter is worth running.
      if (wordsOf(tail).length >= 3 || (armed && tail)) {
        phraseStart.current = now - 250;
        wakeArmed.current = armed;
        dispatch(tail, false);
        clearPhrase();
      }
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
      // A session ending mid-phrase is Chrome recycling the connection, not
      // the operator stopping: flush what we have rather than losing it.
      if (endpointTimer.current) endpoint();
      clearPhrase();
      consumed.current = 0;
      // Chrome ends continuous sessions periodically; revive wake-word mode.
      // Kept short - every millisecond here is a millisecond deaf.
      if (wakeRef.current && !stoppingRef.current) {
        restartTimer.current = setTimeout(() => {
          try {
            recognition.start();
          } catch {
            /* already starting */
          }
        }, RESTART_MS);
      }
    };

    recognitionRef.current = recognition;
    return () => {
      stoppingRef.current = true;
      clearTimeout(restartTimer.current);
      clearTimeout(endpointTimer.current);
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
    consumed.current = 0;
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

  return {
    listening,
    wakeMode,
    interim,
    interimStartedAt,
    supported,
    toggle,
    toggleWake,
    start,
    stop,
  };
}
