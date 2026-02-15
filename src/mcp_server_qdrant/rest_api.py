"""
REST API for n8n sync flows.

Thin FastAPI wrapper around QdrantConnector for deterministic
sync operations (store, update, delete). No source-type checks —
n8n is the sync authority for external sources.
"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from mcp_server_qdrant.models import DocumentMetadata
from mcp_server_qdrant.qdrant import QdrantConnector

logger = logging.getLogger(__name__)


# --- Request/Response Models ---


class StoreRequest(BaseModel):
    """Request body for storing a new document."""

    title: str = Field(description="Document title")
    content: str = Field(description="Full text content")
    collection_name: str = Field(description="Target collection")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional metadata: source_type, source_ref, category, tags, etc."
        ),
    )


class UpdateRequest(BaseModel):
    """Request body for updating an existing document."""

    content: str = Field(description="New full text content (replaces existing)")
    collection_name: str = Field(description="Target collection")
    title: str | None = Field(
        default=None,
        description="New title (optional, keeps existing if not provided)",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Metadata to merge with existing",
    )


class SyncResponse(BaseModel):
    """Unified response for all sync operations."""

    success: bool
    message: str
    document_id: str | None = None
    chunk_count: int = 0


class DocumentSummary(BaseModel):
    """Summary of a document for list responses."""

    document_id: str
    title: str
    abstract: str | None = None
    chunk_count: int = 0
    metadata: DocumentMetadata


class ListResponse(BaseModel):
    """Response for listing documents."""

    success: bool
    count: int
    documents: list[DocumentSummary]


# --- API Factory ---


def create_rest_api(qdrant_connector: QdrantConnector) -> FastAPI:
    """
    Create the FastAPI app with sync endpoints.

    :param qdrant_connector: Shared QdrantConnector instance.
    :return: FastAPI app.
    """
    app = FastAPI(
        title="MCP-Server-Qdrant Sync API",
        description="REST API for n8n sync flows (external source synchronization)",
        version="1.0.0",
    )

    @app.post("/api/v1/documents", response_model=SyncResponse)
    async def store_document(request: StoreRequest) -> SyncResponse:
        """
        Store a new document from an external source.
        Chunks, embeds, and generates abstract automatically.
        Returns the document_id for future updates/deletes.
        """
        try:
            result = await qdrant_connector.store(
                title=request.title,
                content=request.content,
                metadata=request.metadata,
                collection_name=request.collection_name,
            )
            return SyncResponse(
                success=result.success,
                message=result.message,
                document_id=result.document_id,
                chunk_count=result.count,
            )
        except Exception as e:
            logger.error("Store failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/api/v1/documents/{document_id}", response_model=SyncResponse)
    async def update_document(
        document_id: str, request: UpdateRequest
    ) -> SyncResponse:
        """
        Update an existing document (delete old chunks + re-index).
        Preserves the document_id. No source-type check — n8n is
        the sync authority for external sources.
        """
        try:
            # Get existing document to preserve title if not provided
            doc_map = await qdrant_connector.resolve_documents(
                {"document_id": document_id}, request.collection_name
            )
            if not doc_map:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document '{document_id}' not found in '{request.collection_name}'",
                )

            existing_doc = doc_map[document_id]
            title = request.title or existing_doc.title

            # Merge metadata: existing + new
            existing_meta = existing_doc.metadata.model_dump()
            if request.metadata:
                existing_meta.update(request.metadata)

            # Delete old chunks
            await qdrant_connector.delete_all_chunks(
                request.collection_name, document_id
            )

            # Re-store with same document_id
            result = await qdrant_connector.store(
                title=title,
                content=request.content,
                metadata=existing_meta,
                collection_name=request.collection_name,
                document_id=document_id,
            )
            return SyncResponse(
                success=result.success,
                message=f"Updated document '{title}' with {result.count} chunk(s).",
                document_id=document_id,
                chunk_count=result.count,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Update failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/v1/documents/{document_id}", response_model=SyncResponse)
    async def delete_document(
        document_id: str,
        collection_name: str = Query(description="Collection to delete from"),
    ) -> SyncResponse:
        """
        Delete a document and all its chunks.
        No source-type check — n8n is the sync authority.
        """
        try:
            # Verify document exists
            doc_map = await qdrant_connector.resolve_documents(
                {"document_id": document_id}, collection_name
            )
            if not doc_map:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document '{document_id}' not found in '{collection_name}'",
                )

            existing_doc = doc_map[document_id]

            # Delete all chunks
            await qdrant_connector.delete_all_chunks(
                collection_name, document_id
            )

            return SyncResponse(
                success=True,
                message=f"Deleted document '{existing_doc.title}' and all chunks.",
                document_id=document_id,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Delete failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/documents", response_model=ListResponse)
    async def list_documents(
        collection_name: str = Query(description="Collection to search"),
        source_ref: str | None = Query(default=None, description="Filter by source_ref (exact match)"),
        source_type: str | None = Query(default=None, description="Filter by source_type (e.g. 'trilium')"),
        category: str | None = Query(default=None, description="Filter by category"),
        title: str | None = Query(default=None, description="Filter by title (text match)"),
    ) -> ListResponse:
        """
        List documents matching optional filters.
        Used by n8n to look up documents by source_ref before update/delete.
        Returns document-level summaries (no chunk content).
        """
        try:
            filter_dict: dict[str, Any] = {}
            if source_ref is not None:
                filter_dict["source_ref"] = source_ref
            if source_type is not None:
                filter_dict["source_type"] = source_type
            if category is not None:
                filter_dict["category"] = category
            if title is not None:
                filter_dict["title"] = title

            if not filter_dict:
                # No filter = list all (capped at 100)
                doc_map = await qdrant_connector.list_entries(
                    limit=100, collection_name=collection_name
                )
                documents = [
                    DocumentSummary(
                        document_id=doc.document_id,
                        title=doc.title,
                        abstract=doc.abstract,
                        chunk_count=doc.chunk_count,
                        metadata=doc.metadata,
                    )
                    for doc in doc_map
                ]
            else:
                doc_map = await qdrant_connector.resolve_documents(
                    filter_dict, collection_name
                )
                documents = [
                    DocumentSummary(
                        document_id=doc_id,
                        title=doc.title,
                        abstract=doc.abstract,
                        chunk_count=doc.chunk_count,
                        metadata=doc.metadata,
                    )
                    for doc_id, doc in doc_map.items()
                ]

            return ListResponse(
                success=True,
                count=len(documents),
                documents=documents,
            )
        except Exception as e:
            logger.error("List failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/health")
    async def health_check() -> dict:
        """Simple health check for monitoring."""
        return {"status": "ok"}

    return app
