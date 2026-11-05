"""Replace personal data with placeholders before any text is sent to an LLM.

The LLM only ever needs the product lines. Customer identity is resolved in code from the
email header and never leaves the machine. The placeholder map stays local so a human
reviewer can see what was removed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<![\w.])\+?\d[\d ()\-]{8,}\d")
# "Requested by: X" and friends. Spaces are optional because OCR often drops them
# ("Requestedby:Hassanlqbal"); everything after the label to the end of the line is the name.
LABELLED_NAME = re.compile(
    r"((?i:requested\s*by|attn|attention|contact|buyer|prepared\s*by))\s*[:\-]\s*([A-Z][^\n]{1,40}?)[ \t]*$", re.M)
SIGNOFF = re.compile(r"^(thanks|regards|kind regards|best|cheers|many thanks|best regards)[,!.]?\s*$", re.I)


@dataclass
class Redacted:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original


def redact(text: str, known_names: list[str] | None = None, known_orgs: list[str] | None = None) -> Redacted:
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}

    def sub(kind: str, value: str) -> str:
        for ph, orig in mapping.items():
            if orig == value:
                return ph
        counters[kind] = counters.get(kind, 0) + 1
        ph = f"[{kind}_{counters[kind]}]"
        mapping[ph] = value
        return ph

    out = EMAIL.sub(lambda m: sub("EMAIL", m.group(0)), text)
    out = PHONE.sub(lambda m: sub("PHONE", m.group(0).strip()), out)

    names = set(n for n in (known_names or []) if n)
    # labelled names in documents ("Requested by: X"), which OCR may have misspelled
    for m in LABELLED_NAME.finditer(out):
        names.add(m.group(2).strip())
    # the line(s) right after a sign-off are a name and usually a company
    orgs = set(o for o in (known_orgs or []) if o)
    lines = out.split("\n")
    for i, ln in enumerate(lines):
        if SIGNOFF.match(ln.strip()):
            for j, bucket in ((i + 1, names), (i + 2, orgs)):  # person, then usually their company
                cand = lines[j].strip() if j < len(lines) else ""
                if cand and not cand.startswith("[") and len(cand.split()) <= 5 and not any(c.isdigit() for c in cand):
                    bucket.add(cand)
    # full strings first (longest first)
    for n in sorted(names | orgs, key=len, reverse=True):
        out = re.sub(re.escape(n), sub("ORG" if n in orgs and n not in names else "NAME", n), out)
    # then parts of PERSON names on their own ("Hi Hassan", "Mr Iqbal"). Company names are only
    # replaced whole, so "Kelso Water Services" never turns "water pump" into "[NAME] pump".
    parts = {p for n in names for p in n.split() if len(p) > 2 and p[0].isupper()}
    for part in sorted(parts, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(part)}\b", sub("NAME", part), out)
    # OCR glues words ("Hassanlqbal"), so a capitalised part of a name known from the email
    # header also redacts any token containing it. Case-sensitive and 4+ letters, so ordinary
    # words are not hit.
    header_parts = {p for n in (known_names or []) if n for p in n.split() if len(p) >= 4}
    for part in sorted(header_parts, key=len, reverse=True):
        out = re.sub(rf"[\w'.-]*{re.escape(part)}[\w'.-]*", lambda m: sub("NAME", m.group(0)), out)
    return Redacted(text=out, mapping=mapping)


def leaks(text: str, secrets: list[str]) -> list[str]:
    """Return any of the given values still present in text (used by tests and a runtime assert)."""
    low = text.lower()
    return [s for s in secrets if s and s.lower() in low]
