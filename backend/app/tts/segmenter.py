"""Incremental sentence segmentation for streamed speech.

The vocal engine is fastest when it is handed whole sentences: Kokoro
synthesises one sentence in well under half the time it takes to say it, so
if we can give it sentence *one* while the model is still writing sentence
*two*, the assistant starts talking almost as soon as it has something to say.

That only works if we can decide "this is a complete sentence" from a token
stream, where text arrives a few characters at a time and a trailing "." might
be the end of a sentence, a decimal point, or the middle of "Dr.". This module
does that decision and nothing else, so it can be tested on its own.

Two shapes of answer make that harder than it sounds, and both were costing
the operator seconds of silence:

* **No punctuation to break on.** "A marathon is a long-distance running event
  covering a standard distance of twenty-six miles" has no comma anywhere. If
  the opening fragment can only break at a clause, an answer like that is never
  released until the model has finished writing it - the whole feature does
  nothing, silently. So we fall back to breaking at a *word*, choosing one that
  sounds like a breath rather than a stumble.
* **An opening that is too short.** Kokoro's latency has a cliff just below
  ~32 characters: under it, a fragment takes as long to synthesise as one three
  times the size while returning barely a second of audio to cover the next
  pass. A reply beginning "Certainly, sir." must therefore not ship those
  fifteen characters on their own. See `Settings.speech_first_min_chars` for
  the measurements.
"""

from __future__ import annotations

import re

from ..config import settings

#: Words that end in a period without ending a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "ft",
    "vs", "etc", "eg", "ie", "approx", "no", "fig", "al", "inc", "ltd", "co",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
}

#: Terminal punctuation. Colons and semicolons count: spoken prose pauses
#: there anyway, and breaking on them keeps fragments short.
_TERMINALS = ".!?;:\n"

#: Clause boundaries. Not sentence ends, but places a speaker would pause -
#: good enough to break the opening fragment on so speech can start sooner.
_CLAUSE = re.compile(r"[,;:)\]](?=\s)|\s[-—–](?=\s)")

#: Trailing characters that belong to the sentence that precedes them.
_CLOSERS = "\"')”’]}"

#: Words a speaker never pauses after. When the opening fragment has to be cut
#: at a bare word boundary - no comma, no clause break anywhere in range - the
#: cut lands before one of these rather than after it, so the assistant says
#: "…covering" / "a standard distance of…" instead of "…covering a" / "standard
#: distance of…". Same number of words either way; only one of them sounds like
#: a person drawing breath.
_DANGLING = frozenset(
    """a an the and or but nor so yet of to in on at by for with from into over under
    is are was were am be been being has have had do does did will would can could
    shall should may might must that which who whom whose this these those as than
    if then when while about between across through during without within""".split()
)


def _is_boundary(text: str, index: int, final: bool = False) -> bool:
    """Is `text[index]` the end of a speakable fragment?

    `final` says whether `text` is all we are ever going to get. It matters:
    mid-stream, a "." sitting at the very end of the buffer is undecidable -
    the next delta may turn it into a full stop ("… done. Then") or into the
    middle of a token ("…/a.b" + "c"). Guessing "boundary" there is what
    chops URLs in half, so unless the stream has ended we wait one character.
    """
    char = text[index]
    if char == "\n":
        return True

    if char == ".":
        # 3.5 / 1.000 - a decimal point, not a full stop.
        if index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
            return False
        # "Dr." / "e.g." - an abbreviation, not a full stop.
        head = text[:index]
        word = re.split(r"[\s(\[\"']", head)[-1] if head else ""
        if word.replace(".", "").lower() in _ABBREVIATIONS:
            return False

    # Absorb any closing quotes/brackets that trail the punctuation.
    end = index + 1
    while end < len(text) and text[end] in _CLOSERS:
        end += 1
    # A boundary needs whitespace after it; mid-token punctuation such as
    # "https://x" must not split. At the end of the buffer we only know when
    # the stream itself has ended.
    if end >= len(text):
        return final
    return text[end].isspace()


def _boundary_end(text: str, index: int) -> int:
    """Index just past the fragment ending at `text[index]`, closers included."""
    end = index + 1
    while end < len(text) and text[end] in _CLOSERS:
        end += 1
    return end


class SentenceSegmenter:
    """Accumulates streamed text and releases speakable fragments.

    `push` returns the fragments that became complete with this delta; `drain`
    releases whatever is left when the stream ends.
    """

    def __init__(self, min_chars: int | None = None, max_chars: int | None = None,
                 first_min_chars: int | None = None, first_max_chars: int | None = None) -> None:
        self.min_chars = settings.speech_min_chars if min_chars is None else min_chars
        self.max_chars = settings.speech_max_chars if max_chars is None else max_chars
        self.first_min_chars = (
            settings.speech_first_min_chars if first_min_chars is None else first_min_chars
        )
        self.first_max_chars = (
            settings.speech_first_max_chars if first_max_chars is None else first_max_chars
        )
        self._buf = ""
        #: whether anything has been released yet - the opening fragment is
        #: allowed to break earlier than the rest (see settings.speech_first_*)
        self._opened = False

    @property
    def pending(self) -> str:
        return self._buf

    def reset(self) -> None:
        self._buf = ""
        self._opened = False

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            fragment = self._take()
            if fragment is None:
                break
            out.append(fragment)
        return out

    def drain(self) -> list[str]:
        """Release the tail, splitting it if it is unreasonably long."""
        out: list[str] = []
        while self._buf.strip():
            fragment = self._take(final=True)
            if fragment is None:
                rest, self._buf = self._buf.strip(), ""
                if rest:
                    out.append(rest)
                break
            out.append(fragment)
        self._buf = ""
        return out

    # -- internals ------------------------------------------------------------

    def _take(self, final: bool = False) -> str | None:
        buf = self._buf
        if not buf.strip():
            return None

        for index, char in enumerate(buf):
            if char not in _TERMINALS or not _is_boundary(buf, index, final):
                continue
            end = _boundary_end(buf, index)
            fragment = buf[:end].strip()
            # Too short to be worth its own synthesis pass: let it accumulate,
            # unless the stream has ended and this is all we will ever get.
            #
            # The opening fragment answers to the higher floor. A reply that
            # begins "Certainly, sir." would otherwise ship those fifteen
            # characters on their own - and below the cliff at ~32 characters
            # that costs as long to synthesise as ninety would while returning
            # barely a second of audio, so the fragment behind it has no time
            # to be made and the answer stalls audibly right after hello.
            floor = self.min_chars if self._opened else self.first_min_chars
            # At the end of the stream a short fragment is only worth shipping
            # alone if nothing follows it; otherwise fold it into the next one.
            if len(fragment) < floor and (not final or buf[end:].strip()):
                continue
            self._buf = buf[end:]
            self._opened = True
            return fragment or None

        # Nothing has been spoken yet and we already have a clause worth of
        # text: break there rather than making the operator wait for the full
        # stop. Only ever done once per utterance.
        if not self._opened and len(buf) >= self.first_min_chars:
            clause = self._clause_cut(buf)
            if clause is not None:
                self._buf = buf[clause:]
                self._opened = True
                return buf[:clause].strip() or None

        # No terminal punctuation, but the fragment has outgrown the budget:
        # break at the last soft boundary so speech does not stall.
        if len(buf) >= self.max_chars:
            window = buf[: self.max_chars]
            cut = max(window.rfind(", "), window.rfind("; "), window.rfind(" - "))
            if cut < self.min_chars:
                cut = window.rfind(" ")
            if cut > self.min_chars:
                self._buf = buf[cut + 1:]
                self._opened = True
                return buf[: cut + 1].strip() or None

        if final and buf.strip():
            self._buf = ""
            self._opened = True
            return buf.strip()
        return None

    @property
    def _patience(self) -> int:
        """How much text we will wait through for a natural break.

        Two windows, not one. A clause boundary is worth holding out for a
        little longer than the fragment we would otherwise force, and waiting
        this long also settles the question the segmenter cannot otherwise
        answer: whether more text is even coming. An answer that ends inside
        this window was short enough to speak whole, and cutting it would have
        bought a fraction of a second at the cost of an audible seam.
        """
        return self.first_max_chars + self.first_min_chars

    def _clause_cut(self, buf: str) -> int | None:
        """Index to break the opening fragment at, or None to keep waiting."""
        for match in _CLAUSE.finditer(buf[: self._patience]):
            end = match.end()
            if end >= self.first_min_chars:
                return end
        # No clause boundary anywhere in range. Plenty of perfectly ordinary
        # answers contain no comma at all - "A marathon is a long-distance
        # running event covering a standard distance of twenty-six miles" - and
        # holding out for one means waiting for the whole answer to be written
        # before saying a single word, which is exactly what streamed speech
        # exists to avoid. Break on a word boundary instead.
        if len(buf) >= self._patience:
            return self._word_cut(buf[: self.first_max_chars])
        return None

    def _word_cut(self, window: str) -> int | None:
        """Best word boundary in `window` to break the opening fragment at.

        Ranked by how much it sounds like someone drawing breath rather than
        losing their place. The best cut both ends on a real word and leaves
        the next phrase to open on its own function word - "…event covering /
        a standard distance of…". Failing that, any cut not ending on a
        dangling word will do; failing even that, the last one, because a
        slightly awkward join is still better than saying nothing at all until
        the model has finished writing.
        """
        best: int | None = None
        decent: int | None = None
        last: int | None = None

        cut = window.rfind(" ")
        while cut >= self.first_min_chars:
            if last is None:
                last = cut
            before = self._word_at(window[:cut], -1)
            after = self._word_at(window[cut:], 0)
            if before not in _DANGLING:
                if after in _DANGLING:
                    best = cut
                    break
                if decent is None:
                    decent = cut
            cut = window.rfind(" ", 0, cut)

        return best if best is not None else (decent if decent is not None else last)

    @staticmethod
    def _word_at(text: str, index: int) -> str:
        """The first or last bare word of `text`, lowercased."""
        words = text.split()
        if not words:
            return ""
        return words[index].strip(_CLOSERS + "\"'(.,;:!?-").lower()
