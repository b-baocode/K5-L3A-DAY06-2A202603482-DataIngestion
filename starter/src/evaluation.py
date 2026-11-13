"""Evaluate retrieved evidence automatically and keep per-query traces.

Automatic labels only: Hit@1 / Evidence Hit@3 (gold page + all gold quote strings in ONE chunk),
page hits and filter compliance. Coherence and grounding are human labels (see results/audit_*.csv).
"""
import re
from collections import defaultdict
from typing import Any


def canon(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def _quotes(entry: dict[str, Any]) -> list[str]:
    q = entry["evidence_quote"]
    return q if isinstance(q, list) else [q]


def gold_entries(question: dict[str, Any]) -> list[dict[str, Any]]:
    main = {"gold_page": question["gold_page"], "evidence_quote": question["evidence_quote"],
            "gold_source_id": question["gold_source_id"]}
    alts = [{**a, "gold_source_id": a.get("gold_source_id", question["gold_source_id"])}
            for a in question.get("alternative_gold", [])]
    return [main, *alts]


def _hit_detail(question: dict[str, Any], hit: dict[str, Any]) -> dict[str, Any]:
    text = canon(hit["text"])
    best = {"evidence_hit": False, "page_hit": False, "partial_evidence": False}
    for g in gold_entries(question):
        same_page = hit["document_id"] == g["gold_source_id"] and hit["page"] == g["gold_page"]
        quotes = [canon(q) for q in _quotes(g)]
        full = all(q in text for q in quotes)
        # partial: page is right but only a fragment of the evidence is in this chunk (split evidence)
        partial = any(q in text or q[:40] in text or q[-40:] in text for q in quotes) and not full
        best["page_hit"] |= same_page
        best["evidence_hit"] |= same_page and full
        best["partial_evidence"] |= same_page and partial
    return best


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]],
                   applied_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    details = [_hit_detail(question, h) for h in hits]
    req_filter = question.get("filter") or None
    compliant = None
    if req_filter:
        compliant = all(all(h["metadata"].get(k) == v for k, v in req_filter.items()) for h in hits)
    return {
        "id": question["id"], "split": question["split"], "type": question["type"],
        "question": question["question"], "expected_answer": question["expected_answer"],
        "gold_page": question["gold_page"],
        "alternative_gold_pages": [a["gold_page"] for a in question.get("alternative_gold", [])],
        "evidence_quote": question["evidence_quote"],
        "question_filter": req_filter, "applied_filter": applied_filter,
        "hit_at_1": bool(details and details[0]["evidence_hit"]),
        "hit_at_3": any(d["evidence_hit"] for d in details),
        "page_hit_at_3": any(d["page_hit"] for d in details),
        "evidence_split_on_gold_page": any(d["partial_evidence"] for d in details)
                                        and not any(d["evidence_hit"] for d in details),
        "filter_compliant_top3": compliant,
        "results": [
            {"rank": i + 1, "chunk_id": h["chunk_id"], "document_id": h["document_id"], "source": h["source"],
             "page": h["page"], "score": round(h["score"], 4), "metadata": h["metadata"], "text": h["text"],
             **d}
            for i, (h, d) in enumerate(zip(hits, details))
        ],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per strategy and split, as numerator/denominator plus rate."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        groups[(t["strategy"], t["split"])].append(t)

    def frac(n: int, d: int) -> dict[str, Any]:
        return {"n": n, "N": d, "rate": round(n / d, 3) if d else None}

    out: dict[str, Any] = {}
    for (strategy, split), ts in sorted(groups.items()):
        with_filter = [t for t in ts if t["question_filter"]]
        out.setdefault(strategy, {})[split] = {
            "hit_at_1": frac(sum(t["hit_at_1"] for t in ts), len(ts)),
            "evidence_hit_at_3": frac(sum(t["hit_at_3"] for t in ts), len(ts)),
            "page_hit_at_3": frac(sum(t["page_hit_at_3"] for t in ts), len(ts)),
            "evidence_split_on_gold_page": sum(t["evidence_split_on_gold_page"] for t in ts),
            "filter_questions_hit_at_3": frac(sum(t["hit_at_3"] for t in with_filter), len(with_filter)),
            "filter_compliance_top3": frac(sum(bool(t["filter_compliant_top3"]) for t in with_filter),
                                           len(with_filter)) if with_filter else "N/A",
            "mean_latency_ms": round(sum(t["latency_ms"] for t in ts) / len(ts), 1),
        }
    return out
