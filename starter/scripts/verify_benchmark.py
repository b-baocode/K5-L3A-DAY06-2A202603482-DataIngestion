"""Verify that every gold evidence quote exists on its gold page after normalization.

Run from starter/:  python scripts/verify_benchmark.py
Exit code 1 if any check fails. Writes results/benchmark_verification.json.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
from src.ingestion import load_pdf  # noqa: E402


def canon(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def quotes(entry) -> list[str]:
    q = entry["evidence_quote"]
    return q if isinstance(q, list) else [q]


def main() -> int:
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    bench = json.loads((ROOT / "benchmark/questions.json").read_text(encoding="utf-8"))
    src = manifest["sources"][0]
    pages = load_pdf(ROOT / src["local_path"], src["source_id"], src["local_path"])
    included = set(src["included_pages"])
    text = {p.page: canon(p.text) for p in pages}

    errors, report = [], []
    for q in bench["questions"]:
        golds = [{"gold_page": q["gold_page"], "evidence_quote": q["evidence_quote"]}]
        golds += q.get("alternative_gold", [])
        for g in golds:
            page = g["gold_page"]
            found = all(canon(s) in text.get(page, "") for s in quotes(g))
            also_on = sorted(
                p for p in included if p != page and all(canon(s) in text[p] for s in quotes(g))
            )
            report.append({"id": q["id"], "gold_page": page, "found": found,
                           "in_included_pages": page in included, "also_found_on": also_on})
            if not found:
                errors.append(f"{q['id']}: evidence not found on page {page}")
            if page not in included:
                errors.append(f"{q['id']}: gold page {page} is not in included_pages")
        if q.get("filter"):
            ip = q["filter"]["ip_type"]
            if ip not in src["page_metadata"]["ip_type_values"]:
                errors.append(f"{q['id']}: filter value {ip} not in manifest")
            pages_for = src["page_metadata"]["ip_type_pages"][ip]
            if q["gold_page"] not in range(pages_for[0], pages_for[1] + 1):
                errors.append(f"{q['id']}: gold page outside filter range {ip}")

    qs = bench["questions"]
    split = Counter(q["split"] for q in qs)
    filt = Counter(q["split"] for q in qs if q.get("filter"))
    ids = [q["id"] for q in qs]
    if len(qs) < 20: errors.append(f"only {len(qs)} questions (<20)")
    if split["dev"] < 15 or split["test"] < 5: errors.append(f"split {dict(split)} below 15/5")
    if sum(filt.values()) < 3 or filt["test"] < 1: errors.append(f"filter questions {dict(filt)}")
    if len(set(ids)) != len(ids): errors.append("duplicate ids")
    if len(included) < 30: errors.append(f"only {len(included)} included pages (<30)")

    summary = {"questions": len(qs), "split": dict(split), "filter_questions": dict(filt),
               "types": dict(Counter(q["type"] for q in qs)), "included_pages": len(included),
               "errors": errors, "checks": report}
    out = ROOT / "starter/results/benchmark_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in report:
        flag = "OK " if r["found"] and r["in_included_pages"] else "ERR"
        extra = f" (also on {r['also_found_on']})" if r["also_found_on"] else ""
        print(f"[{flag}] {r['id']} p{r['gold_page']}{extra}")
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
