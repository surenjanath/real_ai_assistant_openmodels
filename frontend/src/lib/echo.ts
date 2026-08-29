/**
 * What J.A.R.V.I.S. has recently said out loud, and whether a transcript is
 * just that coming back through the microphone.
 *
 * The acoustic guard in `MicMonitor` is the primary defence and the accurate
 * one. This is the backstop for the cases it cannot cover - headphones
 * unplugged mid-answer, echo cancellation unavailable, a second machine in
 * the room playing the same stream - where the only remaining evidence is
 * that we are being told our own words back.
 *
 * Mirrors `backend/app/echoguard.py`; keep the two in step.
 */

/** How long a spoken phrase stays worth comparing against, in ms. */
const WINDOW_MS = 20_000;
/** Fraction of a transcript's content words that must have just been spoken. */
const THRESHOLD = 0.6;
/** Below this many words a transcript is too short to judge on content. */
const MIN_WORDS = 3;

const STOPWORDS = new Set(
  `a an the and or but if of to in on at for with from by is are was were be been am
   it its this that these those i you he she they we me my your our their as so not no
   do does did done have has had will would can could should may might must there here
   what which who whom how when where why`.split(/\s+/),
);

const spoken: Array<{ at: number; words: Set<string> }> = [];

function words(text: string): string[] {
  return (text ?? "").toLowerCase().match(/[a-z0-9']+/g) ?? [];
}

/** Record something the assistant said aloud. */
export function rememberSpoken(text: string): void {
  const list = words(text);
  if (list.length === 0) return;
  spoken.push({ at: Date.now(), words: new Set(list) });
  if (spoken.length > 32) spoken.splice(0, spoken.length - 32);
}

/** Forget everything - a new conversation shares no vocabulary with the last. */
export function clearSpoken(): void {
  spoken.length = 0;
}

/**
 * Is `text` substantially made of words the assistant just spoke?
 *
 * Containment rather than similarity: the microphone catches a fragment of a
 * long answer, so the transcript being wholly inside what we said is the
 * signal - the reverse is not expected to hold.
 */
export function looksLikeEcho(text: string): boolean {
  const list = words(text);
  if (list.length < MIN_WORDS) return false; // "stop", "louder", "thank you"
  const content = list.filter((word) => !STOPWORDS.has(word));
  if (content.length === 0) return false;

  const cutoff = Date.now() - WINDOW_MS;
  const recent = new Set<string>();
  for (const entry of spoken) {
    if (entry.at < cutoff) continue;
    for (const word of entry.words) recent.add(word);
  }
  if (recent.size === 0) return false;

  let hits = 0;
  for (const word of content) if (recent.has(word)) hits++;
  return hits / content.length >= THRESHOLD;
}
