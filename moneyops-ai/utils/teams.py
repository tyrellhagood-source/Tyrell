"""Team-name normalisation and matching between providers."""
from __future__ import annotations

import difflib
import re
import unicodedata

_ALIASES = {
    "la clippers": "los angeles clippers",
    "la lakers": "los angeles lakers",
    "st louis": "saint louis",
    "man united": "manchester united",
    "man city": "manchester city",
}


def normalize_team(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    s = re.sub(r"\bst\.?\b", "st", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\b(fc|cf|afc|sc)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _ALIASES.get(s, s)


def match_team(name: str, candidates: list[str], threshold: float = 0.86) -> tuple[str | None, float]:
    """Return (best candidate, similarity). None when below threshold."""
    target = normalize_team(name)
    best, best_score = None, 0.0
    for cand in candidates:
        norm = normalize_team(cand)
        if norm == target:
            return cand, 1.0
        score = difflib.SequenceMatcher(None, target, norm).ratio()
        if target and norm and (target.startswith(norm + " ") or norm.startswith(target + " ")):
            score = max(score, 0.9)
        if score > best_score:
            best, best_score = cand, score
    if best_score >= threshold:
        return best, round(best_score, 3)
    return None, round(best_score, 3)
