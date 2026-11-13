"""Read and normalize source data."""
import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

log = logging.getLogger(__name__)

# Running headers/footers of the OpenStax PDF, e.g.
#   "134 4 • Trademark Basics", "4.2 • Early Trademark Systems 135", "Access for free at openstax.org"
_FOOTER_RE = re.compile(
    r"^(?:\d+\s+)?\d+(?:\.\d+)?\s+•\s+.+?(?:\s+\d+)?$|^Access for free at openstax\.org$"
)
MIN_PAGE_CHARS = 200  # below this a page is logged as empty / near-empty


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace, keep line structure, drop running headers/footers."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("­", "")  # soft hyphen
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t ]+", " ", line).strip()
        if not line or _FOOTER_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def printed_page_number(raw_text: str) -> int | None:
    """Return the page number printed in the book's running header/footer, if any."""
    for line in reversed((raw_text or "").splitlines()):
        line = unicodedata.normalize("NFKC", line).strip()
        m = re.match(r"^(\d+)\s+\d+\s+•", line) or re.search(r"•.*\s(\d+)$", line)
        if m:
            return int(m.group(1))
    return None


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page (1-based) with normalized text; log empty pages."""
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = normalize_text(raw)
        if len(text) < MIN_PAGE_CHARS:
            log.warning("%s page %d: only %d chars after normalization", source, i, len(text))
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=i,
                text=text,
                metadata={"printed_page": printed_page_number(raw), "chars": len(text)},
            )
        )
    return pages
