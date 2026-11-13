"""Fixed-size and structure-aware chunking, plus page-level metadata annotation.

Both strategies chunk *within* a page so every chunk maps to exactly one physical page.
Limitation (documented in the report): a sentence that crosses a page break is split.
"""
import re
from typing import Any

from .models import Chunk, DocumentPage

# A numbered section heading is accepted only when followed by "Learning Objectives" or a
# figure caption. This rejects the chapter outline lines on the chapter opening page.
_SECTION_RE = re.compile(r"^(\d\.\d{1,2}) ([A-Z][^\n]{3,80})\n(?:Learning Objectives|Figure \d)", re.M)
_SENT_END = re.compile(r"[.!?:;”\"')\]]$")
_BULLET = re.compile(r"^(?:•|[A-D]\. |\(\d\))")


def annotate_pages(pages: list[DocumentPage], ip_type_pages: dict[str, list[int]]) -> None:
    """Set metadata ip_type (from manifest page ranges) and section headings per page.

    metadata["section_at_start"] = section in effect at offset 0 of the page
    metadata["section_starts"]   = [(offset, "4.6", "The Spectrum of Distinctiveness"), ...]
    """
    current: tuple[str, str] | None = None
    for page in sorted(pages, key=lambda p: p.page):
        ip_type = next((k for k, (lo, hi) in ip_type_pages.items() if lo <= page.page <= hi), None)
        if ip_type is None:
            raise ValueError(f"page {page.page} has no ip_type in manifest")
        page.metadata["ip_type"] = ip_type
        page.metadata["section_at_start"] = current
        starts = []
        for m in _SECTION_RE.finditer(page.text):
            current = (m.group(1), m.group(2).strip())
            starts.append((m.start(), *current))
        page.metadata["section_starts"] = starts


def _section_at(page: DocumentPage, offset: int) -> tuple[str | None, str | None]:
    sec = page.metadata.get("section_at_start")
    for start, num, title in page.metadata.get("section_starts", []):
        if start <= offset:
            sec = (num, title)
    return sec if sec else (None, None)


def _make_chunk(page: DocumentPage, strategy: str, idx: int, text: str, start: int,
                extra: dict[str, Any] | None = None, section_offset: int | None = None) -> Chunk:
    section, section_title = _section_at(page, start if section_offset is None else section_offset)
    meta = {
        "printed_page": page.metadata.get("printed_page"),
        "ip_type": page.metadata.get("ip_type"),
        "section": section,
        "section_title": section_title,
        "chunk_index": idx,
        "char_start": start,
        "char_end": start + len(text),
        "strategy": strategy,
    }
    meta.update(extra or {})
    return Chunk(
        chunk_id=f"{page.document_id}:p{page.page:03d}:{strategy}:{idx:03d}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=meta,
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Character windows of `size` with `overlap`, ignoring sentence/paragraph structure."""
    if not 0 <= overlap < size:
        raise ValueError("require 0 <= overlap < size")
    text, chunks, start, idx = page.text, [], 0, 0
    step = size - overlap
    while start < len(text):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(_make_chunk(page, "fixed", idx, piece, start))
            idx += 1
        if start + size >= len(text):
            break
        start += step
    return chunks


def _paragraph_units(text: str) -> list[tuple[int, str, bool]]:
    """Rebuild paragraphs from layout lines. Returns (offset, text, is_heading)."""
    lines, pos = [], 0
    for line in text.split("\n"):
        lines.append((pos, line))
        pos += len(line) + 1
    width = max((len(l) for _, l in lines), default=0)
    units: list[tuple[int, str, bool]] = []
    buf: list[str] = []
    buf_start = 0

    def flush() -> None:
        nonlocal buf
        if buf:
            units.append((buf_start, " ".join(buf), False))
            buf = []

    for off, line in lines:
        short = len(line) < 0.75 * width
        is_heading = short and not _SENT_END.search(line) and not buf and len(line) < 80 \
            and not _BULLET.match(line)
        if _BULLET.match(line):
            flush()
        if is_heading:
            units.append((off, line, True))
            continue
        if not buf:
            buf_start = off
        buf.append(line)
        if short and _SENT_END.search(line):
            flush()
    flush()
    return units


def _split_long(text: str, max_size: int) -> list[str]:
    """Split an oversized paragraph at sentence boundaries (hard cut as last resort)."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z“\"(])", text)
    parts, cur = [], ""
    for s in sentences:
        while len(s) > max_size:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(s[:max_size])
            s = s[max_size:]
        if cur and len(cur) + 1 + len(s) > max_size:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group reconstructed paragraphs up to `max_size` chars.

    A new numbered section always starts a new chunk; a sub-heading is kept with the
    paragraph that follows it; oversized paragraphs are split at sentence boundaries.
    """
    section_offsets = {s for s, _, _ in page.metadata.get("section_starts", [])}
    groups: list[tuple[list[tuple[int, str, bool]], bool]] = []  # (items, starts_new_section)
    cur: list[tuple[int, str, bool]] = []  # (offset, text, is_heading)
    cur_new_section = False

    def emit(items: list[tuple[int, str, bool]]) -> None:
        nonlocal cur_new_section
        groups.append((items, cur_new_section))
        cur_new_section = False

    for off, unit, is_heading in _paragraph_units(page.text):
        if off in section_offsets:
            if cur:
                emit(cur)
                cur = []
            cur_new_section = True
        pieces = _split_long(unit, max_size) if len(unit) > max_size else [unit]
        search_from = 0
        for piece in pieces:
            # a unit joins its layout lines with " " (same length as "\n"), so the piece's position
            # inside the unit maps 1:1 to the page text
            rel = unit.find(piece, search_from)
            piece_off = off + max(rel, 0)
            search_from = max(rel, 0) + len(piece)
            size = sum(len(t) + 1 for _, t, _ in cur)
            if cur and size + len(piece) > max_size:
                # keep trailing heading(s) together with the paragraph that follows
                carry = []
                while cur and cur[-1][2]:
                    carry.insert(0, cur.pop())
                if cur:
                    emit(cur)
                cur = carry
            cur.append((piece_off, piece, is_heading))
    if cur:
        emit(cur)

    # Merge fragments shorter than MIN_CHUNK: into the previous group of the same section if the
    # result stays <= max_size, otherwise into the following group (never past max_size + MIN_CHUNK).
    def size_of(items: list[tuple[int, str, bool]]) -> int:
        return sum(len(t) + 1 for _, t, _ in items)

    merged: list[list[tuple[int, str, bool]]] = []
    pending: list[tuple[int, str, bool]] = []
    for items, new_section in groups:
        # a fragment at the very top of the page (e.g. "Introduction") joins the next group
        # even across a section start; otherwise a section start keeps fragments apart
        joins = pending and (not new_section or not merged)
        items = pending + list(items) if joins else list(items)
        if pending and not joins:
            merged.append(pending)
        pending = []
        if size_of(items) < MIN_CHUNK:
            if merged and not new_section and size_of(merged[-1]) + size_of(items) <= max_size:
                merged[-1].extend(items)
            else:
                pending = items
            continue
        merged.append(items)
    if pending:
        if merged and size_of(merged[-1]) + size_of(pending) <= max_size + MIN_CHUNK:
            merged[-1].extend(pending)
        else:
            merged.append(pending)

    chunks: list[Chunk] = []
    for items in merged:
        text = "\n".join(t for _, t, _ in items).strip()
        if text:
            # section label: the section heading inside the chunk if it has one, else the one in effect
            sec_off = max((o for o, _, _ in items if o in section_offsets), default=items[0][0])
            chunks.append(_make_chunk(page, "paragraph", len(chunks), text, items[0][0], section_offset=sec_off))
    return chunks


MIN_CHUNK = 150
