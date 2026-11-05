import pytest

from quotedraft.ingest import ingest, parse_email
from quotedraft.redact import leaks, redact
from quotedraft.text import tokenize

from .conftest import DATA


def test_parse_email_header():
    name, email, subject, body = parse_email("From: Sara Whitlock <s.whitlock@kelso-water.example>\nSubject: RFQ\n\nHello")
    assert (name, email, subject, body) == ("Sara Whitlock", "s.whitlock@kelso-water.example", "RFQ", "Hello")


def test_redaction_removes_contact_details():
    raw = "Please call me on +44 7700 912345 or mail joanna@brightwell-eng.example\n\nRegards,\nJoanna Prys\nBrightwell Engineering"
    r = redact(raw, known_names=["Joanna Prys"])
    assert not leaks(r.text, ["+44 7700 912345", "joanna@brightwell-eng.example", "Joanna Prys", "Joanna", "Brightwell Engineering"])
    assert "[EMAIL_1]" in r.text and "[PHONE_1]" in r.text
    assert r.mapping["[EMAIL_1]"] == "joanna@brightwell-eng.example"


@pytest.mark.parametrize("rid", [f"rfq_{i:02d}" for i in range(1, 26) if i not in (1, 4, 5)])  # skip some scans for speed
def test_every_rfq_redacts_identity_but_keeps_product_lines(rid, labels):
    gold = labels[rid]
    att = DATA / "rfqs" / gold["attachment"] if gold.get("attachment") and gold["kind"] == "pdf" else None
    if gold["kind"] == "scan":
        pytest.skip("scans are covered by test_scanned_pdf_is_ocrd")
    raw = (DATA / "rfqs" / f"{rid}.txt").read_text(encoding="utf-8")
    doc = ingest(raw, att)
    r = redact(doc.text, known_names=[doc.sender_name])
    assert not leaks(r.text, [doc.sender_email, doc.sender_name])
    toks = set(tokenize(r.text))
    for line in gold["lines"]:
        missing = set(tokenize(line["text"])) - toks
        assert not missing, f"redaction damaged product line {line['text']!r}: lost {missing}"


def test_company_words_do_not_eat_product_words():
    raw = "1. water pump 4 kW cast iron - 2 pcs\n\nThanks,\nSara Whitlock\nKelso Water Services"
    r = redact(raw, known_names=["Sara Whitlock"])
    assert "water pump 4 kW" in r.text
    assert not leaks(r.text, ["Kelso Water Services", "Sara", "Whitlock"])


def test_labelled_name_is_redacted_even_when_ocr_misspells_it():
    r = redact("Ref: RFQ_01 Requested by: Hassan lqbal\n1. water pump 7.5 kW cast iron x250", known_names=["Hassan Iqbal"])
    assert not leaks(r.text, ["Hassan", "lqbal"])
    assert "water pump 7.5 kW cast iron" in r.text


def test_text_pdf_is_read(labels):
    rid = next(k for k, v in labels.items() if v["kind"] == "pdf")
    raw = (DATA / "rfqs" / f"{rid}.txt").read_text(encoding="utf-8")
    doc = ingest(raw, DATA / "rfqs" / labels[rid]["attachment"])
    assert "pdf-text" in doc.sources
    assert labels[rid]["lines"][0]["text"].split()[0].lower() in doc.text.lower()


def test_scanned_pdf_is_ocrd(labels):
    rid = next(k for k, v in labels.items() if v["kind"] == "scan")
    raw = (DATA / "rfqs" / f"{rid}.txt").read_text(encoding="utf-8")
    doc = ingest(raw, DATA / "rfqs" / labels[rid]["attachment"])
    assert "pdf-ocr" in doc.sources
    red = redact(doc.text, known_names=[doc.sender_name]).text
    first, last = doc.sender_name.split()[0], doc.sender_name.split()[-1]
    assert not leaks(red, [doc.sender_name, first, last, last.replace("I", "l")])
    # OCR may drop spaces; every line's numbers must still be there (the LLM restores spacing)
    squashed = doc.text.replace(" ", "").lower()
    for ln in labels[rid]["lines"]:
        assert ln["text"].replace(" ", "").lower()[:12] in squashed or ln["text"].split()[0].lower() in squashed


@pytest.mark.parametrize("text", [
    "Ref:RFQ_01 Requestedby:Hassanlqbal\n1. waterpump7.5kWcastiron x250",
    "Ref: RFQ_01   Requested by: Hassan lqbal\n1. water pump 7.5 kW cast iron x250",
])
def test_ocr_glued_names_are_redacted(text):
    r = redact(text, known_names=["Hassan Iqbal"])
    assert not leaks(r.text, ["hassan", "lqbal", "iqbal"])
    assert "7.5" in r.text and "x250" in r.text
