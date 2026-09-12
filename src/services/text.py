"""
Generic text utilities the semantic chunker and the answer pipeline both depend on.

This used to live inside services/pdf.py, back when the only source of text was a PDF.
It was split out during the e-commerce pivot (see PLAN.md) because it was never actually
PDF-specific: `sentences_with_pages()` operates on a list of {"page": n, "text": "..."}
dicts, regardless of what produced them, and `format_pages()` is a pure string formatter.
PDF EXTRACTION (PyMuPDF, OCR) was the part of the old pdf.py that was genuinely
PDF-specific - that part is gone, not moved, per the "PDF extraction has no place in this
product" decision.

The "page" vocabulary stays even though products/reviews have no pages, because
`chunking.chunk_pages()` and the citation plumbing in ml/llm.py and api/chat.py still
key off it - they are due for a broader rewrite in Phase 4/5 (the agentic router, product
cards) rather than a half-updated rename here.
"""
import re
from typing import Dict, List, Tuple

# Sentence boundary: ., ! or ? followed by whitespace and a capital/quote/digit.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _split_long_paragraph(paragraph: str, max_words: int) -> List[str]:
    """
    Breaks an over-long paragraph into pieces of at most max_words, preferring sentence
    boundaries and falling back to a hard word split only for a single huge sentence.
    """
    pieces: List[str] = []
    buffer: List[str] = []
    buffer_words = 0

    for sentence in _SENTENCE_END.split(paragraph):
        words = sentence.split()
        if not words:
            continue

        if len(words) > max_words:
            # One "sentence" longer than a whole chunk (tables, formula dumps, bad OCR,
            # or - now - a review with no punctuation at all).
            if buffer:
                pieces.append(" ".join(buffer))
                buffer, buffer_words = [], 0
            for start in range(0, len(words), max_words):
                pieces.append(" ".join(words[start:start + max_words]))
            continue

        if buffer_words + len(words) > max_words:
            pieces.append(" ".join(buffer))
            buffer, buffer_words = [], 0

        buffer.extend(words)
        buffer_words += len(words)

    if buffer:
        pieces.append(" ".join(buffer))
    return pieces


def sentences_with_pages(pages: List[Dict], max_words: int = 300) -> List[Tuple[str, int]]:
    """
    Flattens pages into [(sentence, page), ...] - the input the semantic chunker works on.

    Paragraph structure is used only as a guard rail: each paragraph is first passed
    through _split_long_paragraph() so a single runaway "sentence" can never exceed
    max_words and blow past the embedding model's window. Everything after that is a real
    sentence.

    Deliberately NOT deduplicated or filtered by length: a two-word heading (or a one-line
    review) is a legitimate sentence and its embedding is exactly the signal that a new
    topic starts here.
    """
    out: List[Tuple[str, int]] = []
    for page in pages:
        for paragraph in page["text"].split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            for piece in _split_long_paragraph(paragraph, max_words):
                for sentence in _SENTENCE_END.split(piece):
                    sentence = sentence.strip()
                    if sentence:
                        out.append((sentence, page["page"]))
    return out


def format_pages(page_start: int, page_end: int) -> str:
    """'page 7' or 'pages 7-8' - chunks routinely straddle a page break."""
    return f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
