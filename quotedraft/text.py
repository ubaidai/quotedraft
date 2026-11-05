"""Text normalisation shared by the catalog index and incoming queries.

The same function runs on both sides, so 'M12x50', 'M12 x 50' and 'm12 X 50mm' all
produce the same tokens. This matters more than the choice of keyword ranker.
"""
from __future__ import annotations

import re

_INCH_FRAC = {"1/2": "0.5", "3/4": "0.75", "1-1/4": "1.25", "1-1/2": "1.5", "1 1/4": "1.25", "1 1/2": "1.5"}


# Trade shorthand -> canonical catalog words. Appended (not replaced) so the original
# tokens still count. Same table the matcher prompt lists. Added after the first
# retrieval eval showed these phrasings missing at rank 1 (see README, "What the eval found").
TRADE_TERMS = [
    (r"\b(\d{1,2})\s*mm\s*(dia|diameter)\b", r"m\1"),
    (r"\bsealed\b", "2rs"),
    (r"\bshielded\b", "zz"),
    (r"\b(caf|fibre|fiber)\b", "compressed fibre"),
    (r"\b(galv|galvanized|galvanised|hdg)\b", "hot-dip galvanised"),
    (r"\b(bzp|zp)\b", "zinc plated"),
    (r"\bss304\b", "a2 stainless"),
    (r"\b(ss316|316)\b", "stainless steel 316"),
    (r"\bscrewed\b", "bsp threaded"),
    (r"\bjubilee\b", "hose clip worm drive"),
    (r"\bzip ties?\b", "cable ties"),
    (r"\bangle iron\b", "equal angle"),
    (r"\bhhb\b", "hex head bolt"),
    (r"\bff\b", "full face"),
]


def expand_trade_terms(s: str) -> str:
    extra = []
    for pat, canon in TRADE_TERMS:
        for m in re.finditer(pat, s):
            extra.append(m.expand(canon) if "\\" in canon else canon)
    return s + (" " + " ".join(extra) if extra else "")


def normalize(text: str) -> str:
    s = text.lower()
    s = s.replace("×", "x").replace("”", '"').replace("″", '"')
    # inch marks: 1-1/2" / 1 1/2 inch -> 1.5 inch
    for frac, dec in sorted(_INCH_FRAC.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(rf"(?<![\d/.-]){re.escape(frac)}\s*(\"|inch\b|in\b)", f"{dec} inch", s)
    s = re.sub(r"(\d)\s*(\"|inch\b|in\b)", r"\1 inch", s)
    # 50x50x5 / m12x50 -> separate numbers
    s = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " x ", s)
    # 50mm -> 50 mm, 4kw -> 4 kw   (but keep m12, dn50, 6205 intact)
    s = re.sub(r"(?<=\d)(mm|kw|m)\b", r" \1", s)
    s = re.sub(r"[^a-z0-9.\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return expand_trade_terms(s) if EXPAND else s


EXPAND = True  # eval toggles this to measure the before/after


_STOP = {"x", "the", "a", "of", "and", "for", "to", "in", "with", "mm", "pcs", "qty", "off", "please"}


def tokenize(text: str) -> list[str]:
    toks = []
    for t in normalize(text).replace("-", " ").split():
        t = t.strip(".")
        if t and t not in _STOP:
            toks.append(t)
    return toks
