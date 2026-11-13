"""Run the benchmark for strategies A/B/C.

Protocol (run from starter/, after `python -m src.ingest`):
  1. python -m src.run --split dev    -> dev traces + metrics, then results/selection.json
                                         (strategy chosen by the pre-registered criterion)
  2. python -m src.run --split test   -> refuses to run unless selection.json exists;
                                         test traces + metrics, audit sheets, experiment log
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .chunking import annotate_pages  # noqa: F401  (kept for clarity of the pipeline)
from .embeddings import from_config
from .evaluation import evaluate_query, summarize
from .ingest import build_chunks, load_config, load_pages
from .store import PgVectorStore

STARTER = Path(__file__).resolve().parents[1]
ROOT = STARTER.parent
RESULTS = STARTER / "results"

# Fixed before any score was seen: 10 evenly spread content pages, position = middle of the page text.
COHERENCE_PAGES = [1, 5, 9, 13, 17, 21, 32, 36, 40, 44]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_split(split: str, cfg: dict, questions: list[dict], embedder, store) -> list[dict]:
    traces = []
    qs = [q for q in questions if q["split"] == split]
    for q in qs:
        t0 = time.perf_counter()
        qvec = embedder.embed_query(q["question"])
        t_embed = (time.perf_counter() - t0) * 1000
        for name, spec in cfg["strategies"].items():
            applied = q.get("filter") if spec["use_filter"] and q.get("filter") else None
            t = time.perf_counter()
            hits = store.search(spec["index"], qvec, top_k=cfg["top_k"], filters=applied)
            t_search = (time.perf_counter() - t) * 1000
            tr = evaluate_query(q, hits, applied_filter=applied)
            tr.update({"strategy": name, "index": spec["index"], "latency_ms": round(t_embed + t_search, 1),
                       "search_ms": round(t_search, 1),
                       "filter_note": None if applied or not q.get("filter") else
                       ("question has a filter but this strategy does not apply filters" if not spec["use_filter"]
                        else None),
                       "no_filter_question": not q.get("filter")})
            traces.append(tr)
    return traces


def select_strategy(metrics: dict) -> dict:
    """Pre-registered criterion (benchmark/README.md): dev Evidence Hit@3, then dev Hit@1 if within
    one question; remaining ties are left to manual grounding / resources."""
    dev = {s: m["dev"] for s, m in metrics.items()}
    ranked = sorted(dev, key=lambda s: (dev[s]["evidence_hit_at_3"]["n"], dev[s]["hit_at_1"]["n"]), reverse=True)
    best = ranked[0]
    near = [s for s in ranked if dev[best]["evidence_hit_at_3"]["n"] - dev[s]["evidence_hit_at_3"]["n"] <= 1]
    if len(near) > 1:
        top_h1 = max(dev[s]["hit_at_1"]["n"] for s in near)
        near = [s for s in near if dev[s]["hit_at_1"]["n"] == top_h1]
    tied_before_amendment = list(near)
    if len(near) > 1:
        # AMENDMENT (added after seeing dev, before any test run): grounding is only labelled on test,
        # so it cannot break a tie without peeking. Use dev filter compliance instead.
        def compliance(s: str) -> int:
            c = dev[s]["filter_compliance_top3"]
            return c["n"] if isinstance(c, dict) else -1
        best_c = max(compliance(s) for s in near)
        near = [s for s in near if compliance(s) == best_c]
    return {"selected": near[0], "tied_after_rules": near if len(near) > 1 else [],
            "tied_before_amendment": tied_before_amendment if len(tied_before_amendment) > 1 else [],
            "ranking_dev": ranked,
            "rule": "max dev Evidence Hit@3; if within 1 question -> max dev Hit@1; "
                    "AMENDMENT (pre-test): then max dev filter compliance; then resources",
            "dev_scores": {s: {"hit_at_3": dev[s]["evidence_hit_at_3"], "hit_at_1": dev[s]["hit_at_1"],
                               "filter_compliance": dev[s]["filter_compliance_top3"]} for s in ranked},
            "decided_at": now(), "decided_before_test_run": True}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def audit_sheets(cfg: dict, test_traces: list[dict]) -> None:
    """Blank manual-label sheets. Labels must be filled by the student, not generated."""
    # 1. Coherence: same 10 source positions for every strategy (C uses B's chunks -> same chunk).
    pages = {p.page: p for p in load_pages()}
    rows = []
    for pid, pg in enumerate(COHERENCE_PAGES, 1):
        page = pages[pg]
        off = len(page.text) // 2
        while off < len(page.text) and page.text[off].isspace():  # never sit on a boundary newline
            off += 1
        for name, spec in cfg["strategies"].items():
            # chunks of a page are contiguous and ordered: the covering chunk is the last one starting
            # at or before the offset (for A with overlap, that is the later of two covering windows)
            c = [c for c in build_chunks([page], spec) if c.metadata["char_start"] <= off][-1]
            rows.append({"position_id": f"pos{pid:02d}", "page": pg, "char_offset": off,
                         "source_context": page.text[max(0, off - 60):off + 60].replace("\n", " "),
                         "strategy": name, "same_chunk_as": "B" if spec["index"] == "B" and name != "B" else "",
                         "chunk_id": c.chunk_id, "chunk_text": c.text,
                         "coherence_0_2": "", "reason": ""})
    write_csv(RESULTS / "audit_coherence.csv", rows)

    # 2. Grounding: 5 test questions x strategies, judge the top-3 together.
    rows = []
    for t in test_traces:
        row = {"id": t["id"], "strategy": t["strategy"], "question": t["question"],
               "applied_filter": json.dumps(t["applied_filter"]) if t["applied_filter"] else "",
               "expected_answer": t["expected_answer"], "auto_hit_at_3": t["hit_at_3"]}
        for r in t["results"]:
            row[f"r{r['rank']}_page"] = r["page"]
            row[f"r{r['rank']}_text"] = r["text"]
        row.update({"grounding_0_2": "", "reason": ""})
        rows.append(row)
    write_csv(RESULTS / "audit_grounding.csv", rows)

    # 3. Metadata: 5 test questions x top-3 x strategies; check each field against the PDF.
    rows = []
    for t in test_traces:
        for r in t["results"]:
            m = r["metadata"]
            rows.append({"id": t["id"], "strategy": t["strategy"], "rank": r["rank"], "chunk_id": r["chunk_id"],
                         "source": r["source"], "page": r["page"], "printed_page": m.get("printed_page"),
                         "ip_type": m.get("ip_type"), "section": m.get("section"),
                         "section_title": m.get("section_title"),
                         "required_filter": json.dumps(t["question_filter"]) if t["question_filter"] else "",
                         "text_start": r["text"][:150].replace("\n", " "),
                         "source_ok": "", "page_ok": "", "ip_type_ok": "", "section_ok": "", "notes": ""})
    write_csv(RESULTS / "audit_metadata.csv", rows)


def full_embed_seconds() -> dict[str, float]:
    """Embed time of the latest ingest run that embedded every chunk (all 'inserted'/'updated')."""
    out: dict[str, float] = {}
    log = RESULTS / "ingest.log"
    if not log.exists():
        return out
    for line in log.read_text(encoding="utf-8").splitlines():
        m = re.search(r"index (\w+): (\d+) chunks, upsert \{'inserted': (\d+), 'updated': (\d+).*embed ([\d.]+)s", line)
        if m and int(m.group(3)) + int(m.group(4)) == int(m.group(2)):
            out[m.group(1)] = float(m.group(5))
    return out


def update_experiment_log(cfg: dict, metrics: dict, ingest: dict) -> None:
    path = ROOT / "artifacts/experiment_log.csv"
    full = full_embed_seconds()
    rows = []
    for name, spec in cfg["strategies"].items():
        m = metrics[name]
        idx = ingest["indexes"][spec["index"]]
        rows.append({
            "strategy": name, "chunking": spec["chunking"],
            "chunk_size": spec.get("size", spec.get("max_size")), "overlap": spec.get("overlap", 0),
            "metadata": "document_id,source,page,printed_page,ip_type,section,section_title",
            "filter": "ip_type (only when question provides it)" if spec["use_filter"] else "none",
            "vectorizer": f"{cfg['embedding']['model']} ({ingest['dimensions']}d, pgvector cosine exact)",
            "hypothesis": spec["hypothesis"],
            "dev_hit1": f"{m['dev']['hit_at_1']['n']}/{m['dev']['hit_at_1']['N']}",
            "dev_hit3": f"{m['dev']['evidence_hit_at_3']['n']}/{m['dev']['evidence_hit_at_3']['N']}",
            "test_hit1": f"{m['test']['hit_at_1']['n']}/{m['test']['hit_at_1']['N']}",
            "test_hit3": f"{m['test']['evidence_hit_at_3']['n']}/{m['test']['evidence_hit_at_3']['N']}",
            "notes": f"{idx['chunks']} chunks; full embed {full.get(spec['index'], 'n/a')}s on CPU"
                     f"{' (shared index B, no extra cost)' if name != spec['index'] else ''}; cost $0",
        })
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--audit-only", action="store_true",
                    help="rebuild the blank audit sheets from existing test traces (no retrieval run)")
    args = ap.parse_args()

    load_dotenv(STARTER / ".env")
    RESULTS.mkdir(exist_ok=True)
    cfg = load_config()
    if args.audit_only:
        traces = [json.loads(l) for l in open(RESULTS / "traces_test.jsonl", encoding="utf-8")]
        audit_sheets(cfg, traces)
        print("audit sheets rebuilt from traces_test.jsonl")
        return
    if not args.split:
        ap.error("--split is required unless --audit-only")
    bench = json.loads((ROOT / "benchmark/questions.json").read_text(encoding="utf-8"))
    selection_path = RESULTS / "selection.json"
    if args.split == "test" and not selection_path.exists():
        sys.exit("Refusing to run test: run --split dev first so the strategy is selected on dev.")

    embedder = from_config(cfg["embedding"])
    store = PgVectorStore(os.environ["DATABASE_URL"], embedder.dimensions, embedder.model_name)
    for spec in cfg["strategies"].values():
        if store.count(spec["index"]) == 0:
            sys.exit(f"index {spec['index']} is empty: run python -m src.ingest first")

    traces = run_split(args.split, cfg, bench["questions"], embedder, store)
    meta = {"run_at": now(), "split": args.split, "model": embedder.model_name, "top_k": cfg["top_k"],
            "benchmark_version": bench["version"]}
    write_jsonl(RESULTS / f"traces_{args.split}.jsonl", [{**meta, **t} for t in traces])
    metrics = summarize(traces)
    (RESULTS / f"metrics_{args.split}.json").write_text(
        json.dumps({**meta, "metrics": metrics}, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.split == "dev":
        sel = select_strategy(metrics)
        selection_path.write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8")
        print("selected on dev:", sel["selected"], "| ties:", sel["tied_after_rules"])
    else:
        dev = json.loads((RESULTS / "metrics_dev.json").read_text(encoding="utf-8"))["metrics"]
        both = {s: {**dev[s], **metrics[s]} for s in metrics}
        ingest = json.loads((RESULTS / "ingest_stats.json").read_text(encoding="utf-8"))
        update_experiment_log(cfg, both, ingest)
        audit_sheets(cfg, traces)

    for s, m in metrics.items():
        x = m[args.split]
        print(f"{s} {args.split}: Hit@1 {x['hit_at_1']['n']}/{x['hit_at_1']['N']} | "
              f"Hit@3 {x['evidence_hit_at_3']['n']}/{x['evidence_hit_at_3']['N']} | "
              f"PageHit@3 {x['page_hit_at_3']['n']}/{x['page_hit_at_3']['N']} | "
              f"filter Hit@3 {x['filter_questions_hit_at_3']['n']}/{x['filter_questions_hit_at_3']['N']} | "
              f"filter compliance {x['filter_compliance_top3'] if isinstance(x['filter_compliance_top3'], str) else str(x['filter_compliance_top3']['n']) + '/' + str(x['filter_compliance_top3']['N'])} | "
              f"split-evidence {x['evidence_split_on_gold_page']} | {x['mean_latency_ms']} ms")


if __name__ == "__main__":
    main()
