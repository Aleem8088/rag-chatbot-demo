"""Load documents and split them into chunks that remember where they came from."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_CHUNK_CHARS = 700


@dataclass(frozen=True)
class Chunk:
    source: str  # file name
    location: str  # "page 3" for PDFs, "section: Returns" for markdown, "" otherwise
    text: str


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _pack(paragraphs: list[str], limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Greedily merge paragraphs up to `limit` characters; split oversized ones by sentence."""
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= limit:
            pieces.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > limit:
                pieces.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append(current)

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        if buffer and len(buffer) + len(piece) + 2 > limit:
            chunks.append(buffer)
            buffer = piece
        else:
            buffer = f"{buffer}\n\n{piece}".strip()
    if buffer:
        chunks.append(buffer)
    return chunks


def chunk_text(source: str, text: str) -> list[Chunk]:
    """Chunk plain text or markdown. Markdown headings become the chunk's location."""
    chunks: list[Chunk] = []
    heading = ""
    section_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(section_lines).strip()
        if not body:
            return
        location = f"section: {heading}" if heading else ""
        for piece in _pack(_split_paragraphs(body)):
            chunks.append(Chunk(source=source, location=location, text=piece))

    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.*\S)\s*$", line)
        if match:
            flush()
            section_lines = []
            heading = match.group(1)
        else:
            section_lines.append(line)
    flush()
    return chunks


def chunk_pdf_bytes(source: str, data: bytes) -> list[Chunk]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    chunks: list[Chunk] = []
    for number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for piece in _pack(_split_paragraphs(page_text)):
            chunks.append(Chunk(source=source, location=f"page {number}", text=piece))
    return chunks


def chunk_file_bytes(name: str, data: bytes) -> list[Chunk]:
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or 'none'}")
    if suffix == ".pdf":
        return chunk_pdf_bytes(name, data)
    return chunk_text(name, data.decode("utf-8", errors="replace"))


def load_directory(directory: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            chunks.extend(chunk_file_bytes(path.name, path.read_bytes()))
    return chunks
