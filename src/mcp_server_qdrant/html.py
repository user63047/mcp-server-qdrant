"""
HTML processing utilities for content ingestion.

Handles HTML-to-plaintext conversion and image reference extraction.
Uses only Python stdlib (no extra dependencies).
"""

import re
from html import unescape
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    """HTMLParser subclass that extracts visible text content."""

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip_depth: int = 0
        # Tags whose content should be ignored
        self._skip_tags = {"script", "style", "head"}
        # Block-level tags that should produce line breaks
        self._block_tags = {
            "p", "div", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6",
            "li", "tr", "blockquote", "pre", "section", "article",
            "header", "footer", "nav", "aside", "figcaption",
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag in self._skip_tags:
            self._skip_depth += 1
        if self._skip_depth == 0 and tag in self._block_tags:
            self._pieces.append("\n")

    def handle_endtag(self, tag: str):
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
        if self._skip_depth == 0 and tag in self._block_tags:
            self._pieces.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        return "".join(self._pieces)


class _ImageExtractor(HTMLParser):
    """HTMLParser subclass that extracts image source URLs."""

    def __init__(self):
        super().__init__()
        self.image_refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag == "img":
            for attr_name, attr_value in attrs:
                if attr_name == "src" and attr_value:
                    self.image_refs.append(attr_value)


def is_html(text: str) -> bool:
    """
    Detect whether text contains HTML markup.

    Simple heuristic: looks for common HTML tags.
    """
    return bool(re.search(r"<(?:p|div|br|h[1-6]|img|a|ul|ol|li|table|pre|code)\b", text, re.IGNORECASE))


def strip_html(html: str) -> str:
    """
    Convert HTML to clean plaintext.

    - Strips all tags, preserving visible text
    - Converts block-level elements to line breaks
    - Decodes HTML entities
    - Collapses excessive whitespace

    :param html: Raw HTML string.
    :return: Clean plaintext.
    """
    if not html or not is_html(html):
        return html or ""

    extractor = _TextExtractor()
    extractor.feed(html)
    text = extractor.get_text()

    # Decode HTML entities (&amp; → &, etc.)
    text = unescape(text)

    # Collapse multiple blank lines into max 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces on a single line
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Strip leading/trailing whitespace overall
    return text.strip()


def extract_image_refs(html: str) -> list[str]:
    """
    Extract image source URLs from HTML.

    Returns a list of src attribute values from <img> tags.
    Useful for future multimodal embedding support.

    :param html: Raw HTML string.
    :return: List of image URLs/paths.
    """
    if not html or not is_html(html):
        return []

    extractor = _ImageExtractor()
    extractor.feed(html)
    return extractor.image_refs
