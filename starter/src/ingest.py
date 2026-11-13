"""Ingest the corpus into pgvector for every chunk index used by the strategies.

Run from starter/:  python -m src.ingest
Writes results/ingest_stats.json and appends to results/ingest.log.
"""
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .chunking import annotate_pages, by_paragraph, fixed_size
from .embeddings import from_config
from .ingestion import load_pdf
from .models import Chunk
from .store import PgVectorStore

STARTER = Path(__file__).resolve().parents[1]
ROOT = STARTER.parent
RESULTS = STARTER / "results"


def setup_logging() -> None:
    RESULTS.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS / "ingest.log", encoding="utf-8")])


def load_config() -> dict:
    return json.loads((STARTER / "config/strategies.json").read_text(encoding="utf-8"))


def load_pages():
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    src = manifest["sources"][0]
    pages = load_pdf(ROOT / src["local_path"], src["source_id"], src["local_path"])
    pages = [p for p in pages if p.page in set(src["included_pages"])]
    if len(pages) < 30:
        raise RuntimeError(f"only {len(pages)} content pages (<30)")
    annotate_pages(pages, src["page_metadata"]["ip_type_pages"])
    return pages


def build_chunks(pages, spec: dict) -> list[Chunk]:
    if spec["chunking"] == "fixed":
        return [c for p in pages for c in fixed_size(p, spec["size"], spec["overlap"])]
    if spec["chunking"] == "paragraph":
        return [c for p in pages for c in by_paragraph(p, spec["max_size"])]
    raise ValueError(spec["chunking"])


def main() -> None:
    setup_logging()
    log = logging.getLogger("ingest")
    load_dotenv(STARTER / ".env")
    cfg = load_config()
    pages = load_pages()
    log.info("loaded %d content pages, %d chars", len(pages), sum(len(p.text) for p in pages))

    t0 = time.perf_counter()
    embedder = from_config(cfg["embedding"])
    model_load_s = time.perf_counter() - t0
    store = PgVectorStore(os.environ["DATABASE_URL"], embedder.dimensions, embedder.model_name)
    if "--reset" in sys.argv:
        log.warning("--reset: dropping table before ingest")
        store.drop_schema()
    store.create_schema()

    # one physical index per distinct chunking spec (C reuses B's index)
    indexes = {}
    for name, spec in cfg["strategies"].items():
        indexes.setdefault(spec["index"], spec)

    stats = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "model": embedder.model_name, "dimensions": embedder.dimensions,
             "query_prompt": embedder.query_prompt, "document_prompt": embedder.document_prompt,
             "max_seq_length_tokens": embedder.max_seq_length, "model_load_seconds": round(model_load_s, 2),
             "content_pages": len(pages), "cost_usd": 0.0, "indexes": {}}
    for index, spec in indexes.items():
        t = time.perf_counter()
        chunks = build_chunks(pages, spec)
        t_chunk = time.perf_counter() - t
        t = time.perf_counter()
        todo = store.changed_chunks(index, chunks)
        vectors = dict(zip([c.chunk_id for c in todo],
                           embedder.embed_documents([c.text for c in todo]))) if todo else {}
        t_embed = time.perf_counter() - t
        t = time.perf_counter()
        upsert = store.upsert(index, chunks, vectors)
        t_store = time.perf_counter() - t
        lengths = [len(c.text) for c in chunks]
        tokens = [embedder.count_tokens(c.text) for c in chunks]
        s = {
            "spec": spec, "chunks": len(chunks), "rows_in_db": store.count(index),
            "chars_total": sum(lengths), "chars_min": min(lengths), "chars_median": statistics.median(lengths),
            "chars_mean": round(statistics.mean(lengths), 1), "chars_max": max(lengths),
            "tokens_total": sum(tokens), "tokens_max": max(tokens),
            "chunks_truncated_over_max_seq": sum(1 for x in tokens if x > embedder.max_seq_length),
            "chunks_without_section": sum(1 for c in chunks if not c.metadata.get("section")),
            "chunks_embedded_this_run": len(todo),
            "seconds": {"chunk": round(t_chunk, 2), "embed": round(t_embed, 2), "store": round(t_store, 2)},
            "upsert": upsert,
        }
        if s["rows_in_db"] != len(chunks):
            raise RuntimeError(f"index {index}: {s['rows_in_db']} rows in DB but {len(chunks)} chunks")
        stats["indexes"][index] = s
        log.info("index %s: %d chunks, upsert %s, embed %.1fs", index, len(chunks), upsert, t_embed)

    out = RESULTS / "ingest_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(RESULTS / "ingest_runs.jsonl", "a", encoding="utf-8") as f:  # history of every run
        f.write(json.dumps({**stats, "reset": "--reset" in sys.argv}, ensure_ascii=False) + "\n")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
