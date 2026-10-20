"""
Cache-safety guards: cheap lexical checks that veto a semantic match.

Sentence embeddings are good at topic and bad at small words that flip
meaning. Measured with all-mpnet-base-v2 (default threshold 0.70):

    "Should I enable 2FA?"        vs "Should I not enable 2FA?"         -> 0.93
    "What does the 10 seat plan cost?" vs "... the 50 seat plan cost?"  -> 0.95
    "How do I enable 2FA?"        vs "How do I disable 2FA?"            -> 0.86

All of these clear the threshold, so a pure-similarity cache would serve
the wrong answer (see eval/results.md). A guard only ever turns a hit into
a miss, so its failure mode is an extra LLM call, never a wrong answer.
That asymmetry is why it's acceptable for them to be blunt.
"""

import re
from collections.abc import Callable

_TOKEN = re.compile(r"[a-z0-9']+")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")  # standalone: "2FA" is not a number

NEGATIONS = frozenset(
    {
        "not", "no", "never", "without", "none", "nothing", "nobody", "neither", "nor",
        "cannot", "cant", "dont", "doesnt", "didnt", "wont", "isnt", "arent", "wasnt",
        "shouldnt", "wouldnt", "couldnt", "havent", "hasnt",
    }
)  # fmt: skip


def _tokens(text: str) -> list[str]:
    return [t.replace("'", "") for t in _TOKEN.findall(text.lower())]


def is_negated(text: str) -> bool:
    return any(t in NEGATIONS for t in _tokens(text))


# Opposing actions: a question about switching something ON must not be
# answered with the cached answer for switching it OFF, and vice versa.
_ON = re.compile(r"\b(enable|enabling|activate|turn(ing)? on|switch on|add|set ?up|start)\b")
_OFF = re.compile(
    r"\b(disable|disabling|deactivate|turn(ing)? off|switch off|remove|stop|delete|cancel)\b"
)


def action_polarity(text: str) -> frozenset[str]:
    lowered = text.lower()
    return frozenset(
        name for name, pattern in (("on", _ON), ("off", _OFF)) if pattern.search(lowered)
    )


def action_mismatch(a: str, b: str) -> bool:
    """True when one side asks to switch something on and the other off."""
    pa, pb = action_polarity(a), action_polarity(b)
    return bool(pa and pb and pa != pb)


def numbers(text: str) -> frozenset[str]:
    return frozenset(n.replace(",", "") for n in _NUMBER.findall(text))


def negation_mismatch(a: str, b: str) -> bool:
    return is_negated(a) != is_negated(b)


def number_mismatch(a: str, b: str) -> bool:
    return numbers(a) != numbers(b)


GUARDS: dict[str, Callable[[str, str], bool]] = {
    "negation": negation_mismatch,
    "numbers": number_mismatch,
    "opposite_action": action_mismatch,
}


def veto_reason(query: str, cached_query: str) -> str | None:
    """Name of the first guard that rejects this pair, or None if it's safe."""
    for name, mismatch in GUARDS.items():
        if mismatch(query, cached_query):
            return name
    return None
