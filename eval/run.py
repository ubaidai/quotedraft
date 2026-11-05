"""Eval harness.

  python -m eval.run --retrieval-only            # no API key needed: vector vs keyword vs hybrid
  python -m eval.run --routing haiku_first       # full pipeline on all 25 RFQs (needs ANTHROPIC_API_KEY)
  python -m eval.run --routing sonnet_only
  python -m eval.run --compare                   # runs haiku_first and sonnet_only and prints both

Results are written to eval/results/<timestamp>-<name>.json. Every number in the README comes from these files.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quotedraft.config import DATA_DIR, get_settings  # noqa: E402
from quotedraft.text import tokenize  # noqa: E402

RESULTS = ROOT / "eval" / "results"


def load_labels() -> dict:
    return json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}-{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- retrieval

def retrieval_eval(k: int = 5) -> dict:
    from quotedraft import text

    res = {}
    for expand in (False, True):
        text.EXPAND = expand
        res["trade_terms_on" if expand else "trade_terms_off"] = _retrieval_once(k)
    text.EXPAND = True
    return {"queries": res["trade_terms_on"]["queries"], "store": get_settings().store,
            "embed_model": get_settings().embed_model, "runs": res}


def _retrieval_once(k: int) -> dict:
    from quotedraft.store import build_store

    store = build_store(get_settings())
    labels = load_labels()
    queries = [(ln["text"], ln["sku"]) for v in labels.values() for ln in v["lines"] if ln["sku"]]
    # RFQs 13-25 were not used to choose the fusion weight or the trade-term table's weight
    heldout = [(ln["text"], ln["sku"]) for rid in sorted(labels)[12:] for ln in labels[rid]["lines"] if ln["sku"]]
    out = {}
    for mode in ("keyword", "vector", "hybrid"):
        r1 = r5 = mrr = 0.0
        misses = []
        t0 = time.perf_counter()
        for q, gold in queries:
            skus = [c.sku for c in store.search(q, k=10, mode=mode)]
            if skus[:1] == [gold]:
                r1 += 1
            if gold in skus[:k]:
                r5 += 1
            else:
                misses.append({"query": q, "gold": gold, "got": skus[:3]})
            if gold in skus:
                mrr += 1 / (skus.index(gold) + 1)
        n = len(queries)
        held = [(q, g) for q, g in heldout if g]
        h1 = sum(1 for q, g in held if store.search(q, k=1, mode=mode)[0].sku == g) / len(held)
        out[mode] = {"recall@1": round(r1 / n, 3), f"recall@{k}": round(r5 / n, 3), "mrr@10": round(mrr / n, 3),
                     "heldout_recall@1": round(h1, 3),
                     "ms_per_query": round((time.perf_counter() - t0) * 1000 / n, 1), "misses": misses}
    return {"queries": len(queries), "heldout_queries": len(heldout), "modes": out}


def print_retrieval(res: dict):
    print(f"\nRetrieval eval: {res['queries']} labelled queries, store={res['store']}, embeddings={res['embed_model']}")
    for name, run in res["runs"].items():
        print(f"\n[{name}]")
        print(f"{'mode':<9}{'recall@1':>10}{'recall@5':>10}{'MRR@10':>9}{'held-out@1':>12}{'ms/q':>8}")
        for mode, m in run["modes"].items():
            print(f"{mode:<9}{m['recall@1']:>10.1%}{m['recall@5']:>10.1%}{m['mrr@10']:>9.3f}"
                  f"{m['heldout_recall@1']:>12.1%}{m['ms_per_query']:>8}")


# --------------------------------------------------------------------------- full pipeline

def _sim(a: str, b: str) -> float:
    """Token overlap, or character similarity with spaces removed, whichever is higher.
    The second catches OCR-glued text ('FB50x8S275') that shares no whole tokens with its label."""
    from difflib import SequenceMatcher

    ta, tb = set(tokenize(a)), set(tokenize(b))
    tok = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    sa, sb = a.lower().replace(" ", ""), b.lower().replace(" ", "")
    return max(tok, SequenceMatcher(None, sa, sb).ratio() - 0.25)


def align(gold: list[dict], pred: list[dict]) -> list[tuple[int, int | None]]:
    used, pairs = set(), []
    for gi, g in enumerate(gold):
        best, bj = 0.0, None
        for pj, p in enumerate(pred):
            if pj in used:
                continue
            s = _sim(g["text"], p["description"])
            if s > best:
                best, bj = s, pj
        if bj is not None and best >= 0.3:
            used.add(bj)
            pairs.append((gi, bj))
        else:
            pairs.append((gi, None))
    return pairs


def score_rfq(gold: dict, sheet: dict) -> dict:
    g, p = gold["lines"], sheet["lines"]
    pairs = align(g, p)
    aligned = [(gi, pj) for gi, pj in pairs if pj is not None]
    r = {"gold_lines": len(g), "pred_lines": len(p), "aligned": len(aligned),
         "qty_ok": 0, "sku_ok": 0, "sku_total": 0, "nomatch_ok": 0, "nomatch_total": 0, "false_match": 0,
         "wrong_lines": 0, "wrong_flagged": 0, "right_lines": 0, "right_flagged": 0, "errors": []}
    for gi, pj in aligned:
        gl, pl = g[gi], p[pj]
        qty_ok = abs(float(pl["quantity"]) - float(gl["quantity"])) < 1e-6
        r["qty_ok"] += qty_ok
        if gl["sku"] is None:
            r["nomatch_total"] += 1
            ok = pl["sku"] is None
            r["nomatch_ok"] += ok
            r["false_match"] += not ok
        else:
            r["sku_total"] += 1
            ok = pl["sku"] == gl["sku"]
            r["sku_ok"] += ok
        line_ok = ok and qty_ok
        if line_ok:
            r["right_lines"] += 1
            # NO_MATCH lines are flagged by design (a person decides); only count flags on correct matches
            if gl["sku"] is not None:
                r["right_flagged"] += pl["flagged"]
        else:
            r["wrong_lines"] += 1
            r["wrong_flagged"] += pl["flagged"]
            r["errors"].append({"gold": gl, "pred": {k: pl[k] for k in ("description", "quantity", "sku", "confidence", "model", "reason")}})
    for gi, pj in pairs:
        if pj is None:
            r["wrong_lines"] += 1
            r["errors"].append({"gold": g[gi], "pred": None})
    extra = [p[j] for j in range(len(p)) if j not in {pj for _, pj in aligned}]
    r["extra_lines"] = len(extra)
    for e in extra:
        r["errors"].append({"gold": None, "pred": {k: e[k] for k in ("description", "quantity", "sku")}})
    r["exact"] = r["wrong_lines"] == 0 and not extra
    return r


def full_eval(routing: str, limit: int | None = None, only: set[str] | None = None) -> dict:
    from quotedraft.pipeline import build_pipeline

    settings = get_settings()
    settings.routing = routing
    pipe = build_pipeline(settings, create_drafts=False)
    labels = load_labels()
    per, failures = {}, {}
    ids = sorted(labels)[: limit or None]
    if only:
        ids = [i for i in ids if i in only]
    for rid in ids:
        gold = labels[rid]
        raw = (DATA_DIR / "rfqs" / f"{rid}.txt").read_text(encoding="utf-8")
        att = DATA_DIR / "rfqs" / gold["attachment"] if gold.get("attachment") else None
        try:
            sheet = pipe.run(raw, att, run_id=f"eval-{routing}-{rid}-{int(time.time())}").model_dump()
        except Exception as exc:  # noqa: BLE001
            failures[rid] = f"{type(exc).__name__}: {exc}"
            print(f"  {rid}: FAILED {failures[rid]}")
            continue
        s = score_rfq(gold, sheet)
        s["sheet"] = sheet  # kept so results can be rescored without new API calls
        s.update(cost_usd=sheet["cost_usd"], latency_ms=sheet["latency_ms"], escalations=sheet["escalations"],
                 kind=gold["kind"], messy=gold["messy"], injection=gold["injection"], total=sheet["total"])
        per[rid] = s
        print(f"  {rid} [{gold['kind']:<4}] lines {s['aligned']}/{s['gold_lines']} sku {s['sku_ok']}/{s['sku_total']}"
              f" exact={s['exact']} ${s['cost_usd']:.4f} {s['latency_ms']}ms esc={s['escalations']}")
    return {"routing": routing, "rfqs": len(ids), "failures": failures, "summary": summarize(per), "per_rfq": per}


def summarize(per: dict) -> dict:
    if not per:
        return {}
    S = lambda k: sum(v[k] for v in per.values())  # noqa: E731
    lat = sorted(v["latency_ms"] for v in per.values())
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    wrong, right = S("wrong_lines"), S("right_lines")
    return {
        "extraction_recall": round(S("aligned") / S("gold_lines"), 3),
        "extraction_precision": round(S("aligned") / max(1, S("pred_lines")), 3),
        "quantity_accuracy": round(S("qty_ok") / max(1, S("aligned")), 3),
        "sku_accuracy": round(S("sku_ok") / max(1, S("sku_total")), 3),
        "no_match_correct": f"{S('nomatch_ok')}/{S('nomatch_total')}",
        "false_matches": S("false_match"),
        "extra_lines": S("extra_lines"),
        "quote_exact_match": round(sum(v["exact"] for v in per.values()) / len(per), 3),
        "errors_flagged_for_review": f"{S('wrong_flagged')}/{wrong}",
        "correct_matches_flagged": f"{S('right_flagged')}/{S('sku_ok')}",
        "cost_per_rfq_usd": round(S("cost_usd") / len(per), 5),
        "total_cost_usd": round(S("cost_usd"), 4),
        "latency_p50_ms": int(statistics.median(lat)),
        "latency_p95_ms": p95,
        "rfqs_with_escalation": f"{sum(1 for v in per.values() if v['escalations'])}/{len(per)}",
        "by_kind": {k: round(sum(v["exact"] for v in per.values() if v["kind"] == k) /
                             max(1, sum(1 for v in per.values() if v["kind"] == k)), 3) for k in ("body", "pdf", "scan")},
    }


def print_full(res: dict):
    print(f"\nFull eval, routing={res['routing']}, {res['rfqs']} RFQs, failures={len(res['failures'])}")
    for k, v in res["summary"].items():
        print(f"  {k:<28} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--routing", default="haiku_first", choices=["haiku_first", "haiku_only", "sonnet_only"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rescore", help="re-score a saved results JSON with the current scorer (no API calls)")
    ap.add_argument("--only", help="comma-separated RFQ ids to run, e.g. rfq_12")
    a = ap.parse_args()
    if a.rescore:
        res = json.loads(Path(a.rescore).read_text(encoding="utf-8"))
        labels = load_labels()
        for rid, old in res["per_rfq"].items():
            s = score_rfq(labels[rid], old["sheet"])
            s["sheet"] = old["sheet"]
            s.update({k: old[k] for k in ("cost_usd", "latency_ms", "escalations", "kind", "messy", "injection", "total")})
            res["per_rfq"][rid] = s
        res["summary"] = summarize(res["per_rfq"])
        print_full(res)
        Path(a.rescore).write_text(json.dumps(res, indent=2), encoding="utf-8")
        return
    if a.only:
        global_only = set(a.only.split(","))
        for routing in (["haiku_first", "sonnet_only"] if a.compare else [a.routing]):
            res = full_eval(routing, None, only=global_only)
            print_full(res)
        return
    if a.retrieval_only:
        res = retrieval_eval()
        print_retrieval(res)
        print(f"saved {save('retrieval', res)}")
        return
    for routing in (["haiku_first", "sonnet_only"] if a.compare else [a.routing]):
        res = full_eval(routing, a.limit)
        print_full(res)
        print(f"saved {save(routing, res)}")


if __name__ == "__main__":
    main()
