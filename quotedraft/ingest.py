"""Get text out of an RFQ: the email itself plus any PDF attachment.

Text PDFs go through pdfplumber. Pages with no text layer (scans) are rendered and OCR'd
locally with RapidOCR (ONNX, no system install) or Tesseract if present. Nothing is sent
to an external service at this stage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Ingested:
    sender_name: str | None
    sender_email: str | None
    subject: str
    text: str  # body + attachment text, unredacted
    sources: list[str] = field(default_factory=list)  # "body", "pdf-text", "pdf-ocr"


_FROM = re.compile(r"^From:\s*(?:\"?([^\"<\n]+?)\"?\s*)?<?([\w.+-]+@[\w.-]+)>?\s*$", re.M)
_SUBJ = re.compile(r"^Subject:\s*(.*)$", re.M)


def parse_email(raw: str) -> tuple[str | None, str | None, str, str]:
    m = _FROM.search(raw)
    name = m.group(1).strip() if m and m.group(1) else None
    email = m.group(2).lower() if m else None
    s = _SUBJ.search(raw)
    subject = s.group(1).strip() if s else ""
    body = re.sub(r"^(From|To|Subject|Date|Cc):.*\n", "", raw, flags=re.M).strip()
    return name, email, subject, body


def _ocr_image(img) -> str:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        RapidOCR = None
    if RapidOCR is not None:
        import numpy as np

        engine = _ocr_engine(RapidOCR)
        result, _ = engine(np.array(img.convert("RGB")))
        if not result:
            return ""
        return _rows_from_boxes(result)
    try:
        import pytesseract

        return pytesseract.image_to_string(img)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("No OCR engine available: pip install rapidocr_onnxruntime, or install Tesseract") from exc


def _rows_from_boxes(result) -> str:
    """Group OCR boxes into text rows by vertical centre, then read each row left to right."""
    import statistics

    boxes = []
    for quad, txt, _conf in result:
        ys = [pt[1] for pt in quad]
        boxes.append(((min(ys) + max(ys)) / 2, min(pt[0] for pt in quad), max(ys) - min(ys), txt))
    tol = 0.6 * statistics.median(b[2] for b in boxes)
    rows: list[list[tuple[float, str]]] = []
    centres: list[float] = []
    for cy, x, _h, txt in sorted(boxes):
        if centres and abs(cy - centres[-1]) <= tol:
            rows[-1].append((x, txt))
        else:
            rows.append([(x, txt)])
            centres.append(cy)
    return "\n".join("   ".join(_unglue(t) for _, t in sorted(r)) for r in rows)


def _unglue(t: str) -> str:
    """OCR often drops spaces: 'valveDN50' -> 'valve DN50', 'DN40fullface' -> 'DN40 fullface'."""
    t = re.sub(r"(?<=[a-z]{3})(?=[A-Z])", " ", t)  # 3+ lowercase first, so 'kW' stays intact
    return re.sub(r"(?<=\d)(?=[a-z]{4,})", " ", t)


_ENGINE = None


def _ocr_engine(cls):
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = cls()
    return _ENGINE


def pdf_text(path: Path) -> tuple[str, str]:
    """Returns (text, source) where source is pdf-text or pdf-ocr."""
    import pdfplumber

    parts, used_ocr = [], False
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if len(t.strip()) < 20:
                img = page.to_image(resolution=200).original
                t = _ocr_image(img)
                used_ocr = True
            parts.append(t)
    return "\n".join(parts).strip(), "pdf-ocr" if used_ocr else "pdf-text"


def ingest(raw_email: str, attachment: Path | None = None) -> Ingested:
    name, email, subject, body = parse_email(raw_email)
    text, sources = body, ["body"]
    if attachment is not None:
        att, src = pdf_text(attachment)
        text = f"{body}\n\n--- ATTACHMENT: {attachment.name} ---\n{att}"
        sources.append(src)
    return Ingested(sender_name=name, sender_email=email, subject=subject, text=text, sources=sources)
