"""Store and search vectors using PostgreSQL + pgvector (exact cosine search, no ANN index)."""
import hashlib
import json
from typing import Any

import psycopg

from .models import Chunk

TABLE = "chunks"
# Metadata fields that may be used as filters; values are always passed as SQL parameters.
FILTERABLE = {"ip_type": "ip_type", "section": "section", "page": "page", "document_id": "document_id"}


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


def content_hash(text: str, model: str) -> str:
    """Hash of model + text: a model change forces re-embedding of every chunk."""
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int, model_name: str) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is empty")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions
        self.model_name = model_name

    def drop_schema(self) -> None:
        """Remove the table (needed when the embedding dimension changes)."""
        with self._connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {TABLE}")

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, connect_timeout=5)

    def create_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    strategy      text    NOT NULL,
                    chunk_id      text    NOT NULL,
                    document_id   text    NOT NULL,
                    source        text    NOT NULL,
                    page          integer NOT NULL,
                    ip_type       text,
                    section       text,
                    text          text    NOT NULL,
                    content_hash  text    NOT NULL,
                    metadata      jsonb   NOT NULL,
                    embedding     vector({self.dimensions}) NOT NULL,
                    updated_at    timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (strategy, chunk_id)
                )""")
            row = conn.execute(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (TABLE,)).fetchone()
            if row and row[0] != self.dimensions:
                raise ValueError(f"table {TABLE} has vector({row[0]}), model gives {self.dimensions}; "
                                 "re-run ingest with --reset")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_filter_idx ON {TABLE} (strategy, ip_type, section)")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: dict[str, list[float]]) -> dict[str, int]:
        """Insert/update by stable (strategy, chunk_id); delete chunks of the same documents that
        no longer exist (obsolete after a source change). Unchanged chunks are left untouched.

        `chunks` is the full current chunk set; `vectors` maps chunk_id -> vector for changed chunks."""
        if any(len(v) != self.dimensions for v in vectors.values()):
            raise ValueError("vector dimension mismatch")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id in batch")
        stats = {"inserted": 0, "updated": 0, "metadata_updated": 0, "unchanged": 0, "deleted": 0}
        with self._connect() as conn, conn.transaction():
            rows = conn.execute(
                f"SELECT chunk_id, content_hash, metadata FROM {TABLE} WHERE strategy = %s", (strategy,)).fetchall()
            existing = {r[0]: r[1] for r in rows}
            existing_meta = {r[0]: r[2] for r in rows}
            for c in chunks:
                h = content_hash(c.text, self.model_name)
                if existing.get(c.chunk_id) == h:
                    if existing_meta[c.chunk_id] != json.loads(json.dumps(c.metadata)):
                        # same text and vector, corrected metadata: update without re-embedding
                        conn.execute(
                            f"UPDATE {TABLE} SET metadata = %s, ip_type = %s, section = %s, page = %s, "
                            f"source = %s, updated_at = now() WHERE strategy = %s AND chunk_id = %s",
                            (json.dumps(c.metadata), c.metadata.get("ip_type"), c.metadata.get("section"),
                             c.page, c.source, strategy, c.chunk_id))
                        stats["metadata_updated"] += 1
                    else:
                        stats["unchanged"] += 1
                    continue
                if c.chunk_id not in vectors:
                    raise ValueError(f"no vector for changed chunk {c.chunk_id}")
                v = vectors[c.chunk_id]
                stats["updated" if c.chunk_id in existing else "inserted"] += 1
                conn.execute(f"""
                    INSERT INTO {TABLE} (strategy, chunk_id, document_id, source, page, ip_type, section,
                                         text, content_hash, metadata, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                    ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id, source = EXCLUDED.source, page = EXCLUDED.page,
                        ip_type = EXCLUDED.ip_type, section = EXCLUDED.section, text = EXCLUDED.text,
                        content_hash = EXCLUDED.content_hash, metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding, updated_at = now()""",
                    (strategy, c.chunk_id, c.document_id, c.source, c.page, c.metadata.get("ip_type"),
                     c.metadata.get("section"), c.text, h, json.dumps(c.metadata), _vec(v)))
            doc_ids = sorted({c.document_id for c in chunks})
            cur = conn.execute(
                f"DELETE FROM {TABLE} WHERE strategy = %s AND document_id = ANY(%s) AND NOT (chunk_id = ANY(%s))",
                (strategy, doc_ids, ids))
            stats["deleted"] = cur.rowcount
        return stats

    def changed_chunks(self, strategy: str, chunks: list[Chunk]) -> list[Chunk]:
        """Chunks whose (model, text) hash differs from what is stored — only these need embedding."""
        with self._connect() as conn:
            existing = dict(conn.execute(
                f"SELECT chunk_id, content_hash FROM {TABLE} WHERE strategy = %s", (strategy,)).fetchall())
        return [c for c in chunks if existing.get(c.chunk_id) != content_hash(c.text, self.model_name)]

    def count(self, strategy: str) -> int:
        with self._connect() as conn:
            return conn.execute(f"SELECT count(*) FROM {TABLE} WHERE strategy = %s", (strategy,)).fetchone()[0]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError("query vector dimension mismatch")
        where, params = ["strategy = %s"], [strategy]
        for key, value in (filters or {}).items():
            if key not in FILTERABLE:
                raise ValueError(f"filter field not allowed: {key}")
            where.append(f"{FILTERABLE[key]} = %s")
            params.append(value)
        sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {TABLE} WHERE {' AND '.join(where)}
            ORDER BY embedding <=> %s::vector, chunk_id
            LIMIT %s"""
        qv = _vec(query_vector)
        with self._connect() as conn:
            rows = conn.execute(sql, [qv, *params, qv, top_k]).fetchall()
        return [{"chunk_id": r[0], "document_id": r[1], "source": r[2], "page": r[3], "text": r[4],
                 "metadata": r[5], "score": float(r[6])} for r in rows]
