"""
Chunking module for splitting documents into embedding-friendly chunks.

Uses a hybrid approach: target size in tokens with natural boundary detection
(paragraph, newline, sentence, word). Chunks overlap by a configurable amount
to preserve context across boundaries.

Token estimation is character-based (~3.3 chars/token for German, ~4 for English).
The ratio is configurable but does not need to be exact — the chunk target size
already includes a buffer to the model's hard context window limit.
"""

import logging
import re

from mcp_server_qdrant.settings import ChunkingSettings

logger = logging.getLogger(__name__)

# Default character-to-token ratio (German ~3.0-3.5, English ~4.0)
DEFAULT_CHARS_PER_TOKEN = 3.3


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate token count from character length."""
    return int(len(text) / chars_per_token)


def tokens_to_chars(tokens: int, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Convert a token count to approximate character count."""
    return int(tokens * chars_per_token)


def find_boundary(text: str, target_pos: int, min_pos: int) -> int:
    """
    Search backwards from target_pos for a natural text boundary.

    Priority: paragraph (\\n\\n) > newline (\\n) > sentence end (. ) > space

    :param text: The full text to search in.
    :param target_pos: The ideal cut position (character index).
    :param min_pos: The earliest acceptable cut position.
    :return: The character index to cut at (exclusive). Returns target_pos
             if no boundary is found (hard cut).
    """
    if target_pos >= len(text):
        return len(text)

    search_region = text[min_pos:target_pos]

    # Priority 1: Paragraph break (\n\n)
    idx = search_region.rfind("\n\n")
    if idx != -1:
        # Cut after the paragraph break
        return min_pos + idx + 2

    # Priority 2: Newline (\n)
    idx = search_region.rfind("\n")
    if idx != -1:
        return min_pos + idx + 1

    # Priority 3: Sentence end (. or ? or ! followed by space or end)
    # Search for the last sentence-ending punctuation
    match = None
    for m in re.finditer(r'[.!?]\s', search_region):
        match = m
    if match:
        return min_pos + match.end()

    # Priority 4: Word boundary (space)
    idx = search_region.rfind(" ")
    if idx != -1:
        return min_pos + idx + 1

    # No boundary found — hard cut at target
    return target_pos


def chunk_text(
    text: str,
    settings: ChunkingSettings | None = None,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> list[str]:
    """
    Split text into chunks using hybrid strategy (target size + natural boundaries).

    Algorithm:
    1. If text fits in one chunk → return as-is
    2. Otherwise, advance to target size in characters
    3. Search backwards (up to ~300 tokens) for a natural boundary
    4. Cut there, then start the next chunk with overlap

    :param text: The text to chunk.
    :param settings: Chunking configuration (chunk_size, chunk_overlap in tokens).
    :param chars_per_token: Character-to-token ratio for estimation.
    :return: List of text chunks.
    """
    if settings is None:
        settings = ChunkingSettings()

    chunk_size_chars = tokens_to_chars(settings.chunk_size, chars_per_token)
    overlap_chars = tokens_to_chars(settings.chunk_overlap, chars_per_token)
    # Minimum boundary search window: 300 tokens worth of characters
    boundary_search_chars = tokens_to_chars(300, chars_per_token)

    # If text fits in one chunk, return as-is
    if len(text) <= chunk_size_chars:
        return [text.strip()] if text.strip() else []

    chunks = []
    pos = 0

    while pos < len(text):
        # Calculate target end position
        target_end = pos + chunk_size_chars

        # If remaining text fits in one chunk, take it all
        if target_end >= len(text):
            chunk = text[pos:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Find a natural boundary to cut at
        min_boundary = max(pos + 1, target_end - boundary_search_chars)
        cut_pos = find_boundary(text, target_end, min_boundary)

        chunk = text[pos:cut_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Next chunk starts with overlap
        pos = cut_pos - overlap_chars
        # Safety: ensure we always advance
        if pos <= (cut_pos - chunk_size_chars):
            pos = cut_pos

    logger.debug(
        "Chunked text (%d chars, ~%d tokens) into %d chunks",
        len(text),
        estimate_tokens(text, chars_per_token),
        len(chunks),
    )

    return chunks
