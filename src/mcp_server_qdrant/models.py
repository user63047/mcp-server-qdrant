"""
Data models for the two-level document/chunk architecture.

Documents are split into chunks that each fit the embedding model's context window.
All chunks of a document share a document_id and carry redundant document-level
metadata to avoid secondary lookups.

Source types distinguish between composed entries (live in Qdrant) and external
sources (indexed from Trilium, PDFs, etc.).
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Known source types for documents."""

    COMPOSED = "composed"
    TRILIUM = "trilium"
    PDF = "pdf"
    PAPERLESS = "paperless"


class DocumentMetadata(BaseModel):
    """
    Metadata stored on every chunk of a document (redundant).
    Corresponds to the 'metadata' field in the Qdrant payload.
    """

    source_type: str = Field(
        default=SourceType.COMPOSED,
        description="Source type: composed, trilium, pdf, paperless, ...",
    )
    source_ref: str | None = Field(
        default=None,
        description="Link/reference to external source. None for composed entries.",
    )
    category: str | None = Field(
        default=None,
        description="Content category (e.g. 'homelab', 'coding').",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of tags.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp (ISO 8601).",
    )
    updated_at: str | None = Field(
        default=None,
        description="Last modification timestamp (ISO 8601).",
    )
    relevance_score: int = Field(
        default=0,
        description="Access tracking score.",
    )
    last_accessed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Last access timestamp (ISO 8601).",
    )


class ChunkPayload(BaseModel):
    """
    The full payload stored on a single Qdrant point (chunk).

    This is a flat structure — document-level fields (document_id, title, abstract)
    are duplicated on every chunk. The 'metadata' sub-object contains all
    document-level metadata.
    """

    document_id: str = Field(description="UUID grouping all chunks of a document.")
    title: str = Field(description="Human-readable document title.")
    chunk_index: int = Field(description="Position within the document (0, 1, 2, ...).")
    abstract: str | None = Field(
        default=None,
        description="LLM-generated summary of the whole document.",
    )
    full_content: str | None = Field(
        default=None,
        description="Full original text. Only on chunk_index=0 for source_type=composed.",
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata,
        description="Document-level metadata (redundant on every chunk).",
    )

    def to_qdrant_payload(self) -> dict[str, Any]:
        """Convert to the dict structure stored in Qdrant point payload."""
        payload: dict[str, Any] = {
            "document_id": self.document_id,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata.model_dump(),
        }
        if self.abstract is not None:
            payload["abstract"] = self.abstract
        if self.full_content is not None:
            payload["full_content"] = self.full_content
        return payload

    @classmethod
    def from_qdrant_payload(cls, payload: dict[str, Any]) -> "ChunkPayload":
        """Reconstruct from a Qdrant point payload dict."""
        return cls(
            document_id=payload["document_id"],
            title=payload["title"],
            chunk_index=payload.get("chunk_index", 0),
            abstract=payload.get("abstract"),
            full_content=payload.get("full_content"),
            metadata=DocumentMetadata(**payload.get("metadata", {})),
        )


class ChunkWithId(BaseModel):
    """A chunk payload together with its Qdrant point ID."""

    point_id: str = Field(description="Qdrant point ID.")
    content: str = Field(description="The chunk text (embedded content).")
    payload: ChunkPayload = Field(description="Full chunk payload.")


class DocumentResult(BaseModel):
    """
    Document-level result returned from search/list operations.
    Chunks are grouped by document_id and deduplicated.

    The chunks field contains the actual chunk texts (the embedded content).
    total_chunk_count is the total number of chunks in the document,
    while len(chunks) may be fewer if not all chunks were loaded (hybrid mode).
    """

    document_id: str
    title: str
    abstract: str | None = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    chunks: list[str] = Field(
        default_factory=list,
        description="Chunk texts included in this result (may be partial).",
    )
    chunk_count: int = Field(
        default=1,
        description="Number of matched/loaded chunks in this result.",
    )
    total_chunk_count: int | None = Field(
        default=None,
        description="Total number of chunks in the document (None if unknown).",
    )

    def format_for_llm(self) -> str:
        """Format this document result for LLM consumption.

        Includes chunk content so the LLM has access to the actual
        information, not just the abstract summary.
        """
        parts = [f"<document id=\"{self.document_id}\">"]
        parts.append(f"<title>{self.title}</title>")
        if self.abstract:
            parts.append(f"<abstract>{self.abstract}</abstract>")

        # Metadata
        meta = self.metadata
        meta_parts = []
        if meta.source_type:
            meta_parts.append(f"source_type={meta.source_type}")
        if meta.source_ref:
            meta_parts.append(f"source_ref={meta.source_ref}")
        if meta.category:
            meta_parts.append(f"category={meta.category}")
        if meta.tags:
            meta_parts.append(f"tags={','.join(meta.tags)}")
        if meta_parts:
            parts.append(f"<metadata>{' | '.join(meta_parts)}</metadata>")

        # Chunk content
        if self.chunks:
            parts.append("<content>")
            for i, chunk_text in enumerate(self.chunks):
                if len(self.chunks) > 1:
                    parts.append(f"--- chunk {i} ---")
                parts.append(chunk_text)
            parts.append("</content>")

        # Hint for partial loading
        if (
            self.total_chunk_count is not None
            and len(self.chunks) < self.total_chunk_count
        ):
            parts.append(
                f"<note>Showing {len(self.chunks)} of {self.total_chunk_count} chunks. "
                f"Use qdrant-find with filter {{\"document_id\": \"{self.document_id}\"}} "
                f"to load the full document.</note>"
            )

        parts.append("</document>")
        return "\n".join(parts)


def generate_document_id() -> str:
    """Generate a new unique document ID."""
    return uuid.uuid4().hex
