"""Is this transcript the assistant hearing its own voice?

On speakers an open microphone captures J.A.R.V.I.S. as well as the operator,
and a speech recogniser will happily transcribe it. Timing alone cannot settle
it: Chrome finalises a phrase a second or more after the audio that produced
it, by which point the utterance it echoed has already stopped playing, so a
"was audio playing *now*" test lets every echo through.

This module is the content half of the defence - it asks whether the words we
just heard are words we just said. Cheap, order-insensitive containment: what
fraction of the transcript's own vocabulary appears in the recent speech. The
mic usually catches a *fragment* of a sentence, so containment (not symmetric
similarity) is the right measure - "the current time is half past four" is a
perfect echo of a much longer utterance.
"""

from __future__ import annotations

import re

#: Words too common to carry any evidence either way. A transcript made only
#: of these can never be judged an echo on content alone.
_STOPWORDS = frozenset(
    """a an the and or but if of to in on at for with from by is are was were be been am
    it its this that these those i you he she they we me my your our their as so not no
    do does did done have has had will would can could should may might must there here
    what which who whom how when where why""".split()
)

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def containment(candidate: str, spoken: str) -> float:
    """Fraction of `candidate`'s content words that appear in `spoken`.

    1.0 means every meaningful word of the transcript was just said aloud by
    the assistant; 0.0 means none of them were.
    """
    cand = [word for word in _tokens(candidate) if word not in _STOPWORDS]
    if not cand:
        return 0.0
    said = set(_tokens(spoken))
    if not said:
        return 0.0
    hits = sum(1 for word in cand if word in said)
    return hits / len(cand)


def is_echo(candidate: str, spoken: str, threshold: float = 0.6,
            min_words: int = 3) -> bool:
    """Judge `candidate` to be the assistant's own voice coming back in.

    `min_words` keeps short genuine directives safe: "stop", "louder", "thank
    you" share vocabulary with almost any answer, and refusing them because
    the assistant happened to say "stop" would be worse than the echo.
    """
    words = _tokens(candidate)
    if len(words) < min_words:
        return False
    return containment(candidate, spoken) >= threshold
