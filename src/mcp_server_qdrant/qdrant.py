import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.chunking import chunk_text
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.models import (
    ChunkPayload,
    ChunkWithId,
    DocumentMetadata,
    DocumentResult,
    SourceType,
    generate_document_id,
)
from mcp_server_qdrant.settings import ChunkingSettings, METADATA_PATH
from mcp_server_qdrant.summary import SummaryProvider

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]

# --- Legacy models (kept for backward compatibility during transition) ---


class Entry(BaseModel):
    """A single entry in the Qdrant collection. DEPRECATED: use models.DocumentResult."""

    content: str
    metadata: Metadata | None = None


class EntryWithId(BaseModel):
    """A single entry with its point ID. DEPRECATED: use models.ChunkWithId."""

    id: str
    content: str
    metadata: Metadata | None = None


# --- Operation result for tools layer ---


class OperationResult(BaseModel):
    """Result of a write operation, supporting disambiguation."""

    success: bool
    message: str
    count: int = 0
    document_id: str | None = None  # returned on store/update for sync consumers
    documents: list[DocumentResult] | None = None  # populated on disambiguation


# --- Connector ---


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server using the two-level
    document/chunk architecture.

    Documents are split into chunks that fit the embedding model's context
    window. All chunks share a document_id and carry redundant metadata.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
        chunking_settings: ChunkingSettings | None = None,
        summary_provider: SummaryProvider | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes
        self._chunking_settings = chunking_settings or ChunkingSettings()
        self._summary_provider = summary_provider

    # =====================================================================
    # Public API
    # =====================================================================

    async def get_collection_names(self) -> list[str]:
        """Get the names of all collections in the Qdrant server."""
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    # ----- store ---------------------------------------------------------

    async def store(
        self,
        title: str,
        content: str,
        metadata: Metadata | None = None,
        *,
        collection_name: str | None = None,
        document_id: str | None = None,
    ) -> OperationResult:
        """
        Store a document: generate abstract, chunk text, embed each chunk,
        and write all points with a shared document_id.

        :param title: Document title.
        :param content: Full document text.
        :param metadata: Optional metadata dict (category, tags, source_type, source_ref, ...).
        :param collection_name: Target collection (uses default if None).
        :param document_id: Optional document_id (used by sync to preserve ID on re-index).
        :return: OperationResult with success info.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        meta = metadata or {}
        now = datetime.now(timezone.utc).isoformat()
        source_type = meta.get("source_type", SourceType.COMPOSED)

        # Build DocumentMetadata
        doc_meta = DocumentMetadata(
            source_type=source_type,
            source_ref=meta.get("source_ref"),
            category=meta.get("category"),
            tags=meta.get("tags", []),
            created_at=now,
            updated_at=None,
            relevance_score=0,
            last_accessed_at=now,
        )

        # Generate abstract
        abstract = None
        if self._summary_provider and self._summary_provider.enabled:
            abstract = await self._summary_provider.generate_abstract(content, title)

        # Chunk the content
        chunks = chunk_text(content, self._chunking_settings)
        if not chunks:
            return OperationResult(success=False, message="Empty content, nothing to store.")

        doc_id = document_id or generate_document_id()

        # Embed all chunks
        embeddings = await self._embedding_provider.embed_documents(chunks)
        vector_name = self._embedding_provider.get_vector_name()

        # Build points
        points = []
        for i, (chunk_text_content, embedding) in enumerate(zip(chunks, embeddings)):
            payload: dict[str, Any] = {
                "document": chunk_text_content,
                "document_id": doc_id,
                "title": title,
                "chunk_index": i,
                METADATA_PATH: doc_meta.model_dump(),
            }
            if abstract is not None:
                payload["abstract"] = abstract
            # full_content only on chunk 0 for composed entries
            if i == 0 and source_type == SourceType.COMPOSED:
                payload["full_content"] = content

            points.append(
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embedding},
                    payload=payload,
                )
            )

        await self._client.upsert(collection_name=collection_name, points=points)

        logger.info(
            "Stored document '%s' (%s) with %d chunk(s) in '%s'",
            title, doc_id, len(chunks), collection_name,
        )
        return OperationResult(
            success=True,
            message=f"Stored document '{title}' with {len(chunks)} chunk(s).",
            count=len(chunks),
            document_id=doc_id,
        )

    # ----- search / find -------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[DocumentResult]:
        """
        Search for documents. Returns deduplicated document-level results
        grouped by document_id.

        :param query: Search query text.
        :param collection_name: Collection to search.
        :param limit: Max number of chunk hits (documents returned may be fewer).
        :param query_filter: Optional Qdrant filter.
        :return: List of DocumentResult.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )

        if not search_results.points:
            return []

        # Group by document_id and deduplicate
        documents = self._group_points_to_documents(search_results.points)

        # Access tracking: update all chunks of found documents (find = +3)
        all_doc_ids = [doc.document_id for doc in documents]
        await self._update_access_tracking_by_document_ids(
            collection_name, all_doc_ids, score_increment=3
        )

        return documents

    # ----- list ----------------------------------------------------------

    async def list_entries(
        self,
        filter_dict: dict | None = None,
        limit: int = 10,
        *,
        collection_name: str | None = None,
    ) -> list[DocumentResult]:
        """
        List documents in a collection, grouped by document_id.

        :param filter_dict: Optional filter criteria.
        :param limit: Max number of documents to return.
        :param collection_name: Collection to list from.
        :return: List of DocumentResult.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        qdrant_filter = self._build_filter(filter_dict) if filter_dict else None

        # Scroll enough chunks to find `limit` distinct documents
        # Over-fetch to account for multi-chunk documents
        fetch_limit = limit * 5
        results, _ = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            return []

        documents = self._group_points_to_documents(results)[:limit]

        # Access tracking (list = +1)
        all_doc_ids = [doc.document_id for doc in documents]
        await self._update_access_tracking_by_document_ids(
            collection_name, all_doc_ids, score_increment=1
        )

        return documents

    # ----- delete --------------------------------------------------------

    async def delete(
        self,
        filter_dict: dict,
        *,
        collection_name: str | None = None,
    ) -> OperationResult:
        """
        Delete documents matching the filter. Only composed entries can be
        deleted; external sources return a redirect message.
        Disambiguates if multiple documents match.

        :param filter_dict: Filter criteria.
        :param collection_name: Collection to delete from.
        :return: OperationResult (may contain disambiguation documents).
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return OperationResult(success=False, message="Collection does not exist.", count=0)

        doc_map = await self.resolve_documents(filter_dict, collection_name)

        if not doc_map:
            return OperationResult(success=False, message="No matching documents found.", count=0)

        # Disambiguation: multiple documents matched
        if len(doc_map) > 1:
            docs = list(doc_map.values())
            return OperationResult(
                success=False,
                message=f"Multiple documents matched ({len(docs)}). Please specify which one by document_id.",
                documents=docs,
            )

        doc_id, doc_result = next(iter(doc_map.items()))

        # Source type check
        if doc_result.metadata.source_type != SourceType.COMPOSED:
            return OperationResult(
                success=False,
                message=(
                    f"Cannot delete '{doc_result.title}' — it is a '{doc_result.metadata.source_type}' entry "
                    f"(source: {doc_result.metadata.source_ref}). "
                    f"Delete it at the source; the sync will update Qdrant automatically."
                ),
                documents=[doc_result],
            )

        # Delete all chunks of this document
        await self.delete_all_chunks(collection_name, doc_id)

        return OperationResult(
            success=True,
            message=f"Deleted document '{doc_result.title}' ({doc_result.chunk_count} chunk(s)).",
            count=1,
        )

    # ----- update --------------------------------------------------------

    async def update(
        self,
        filter_dict: dict,
        new_content: str,
        new_metadata: Metadata | None = None,
        *,
        collection_name: str | None = None,
    ) -> OperationResult:
        """
        Replace the content of a document: delete old chunks, re-chunk,
        re-embed, generate new abstract. Only for composed entries.

        :param filter_dict: Filter to identify the document.
        :param new_content: Replacement text.
        :param new_metadata: Optional metadata updates (merged with existing).
        :param collection_name: Collection.
        :return: OperationResult.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return OperationResult(success=False, message="Collection does not exist.", count=0)

        doc_map = await self.resolve_documents(filter_dict, collection_name)

        if not doc_map:
            return OperationResult(success=False, message="No matching documents found.", count=0)

        if len(doc_map) > 1:
            docs = list(doc_map.values())
            return OperationResult(
                success=False,
                message=f"Multiple documents matched ({len(docs)}). Please specify which one by document_id.",
                documents=docs,
            )

        doc_id, doc_result = next(iter(doc_map.items()))

        if doc_result.metadata.source_type != SourceType.COMPOSED:
            return OperationResult(
                success=False,
                message=(
                    f"Cannot update '{doc_result.title}' — it is a '{doc_result.metadata.source_type}' entry "
                    f"(source: {doc_result.metadata.source_ref}). "
                    f"Edit the source directly; the sync will update Qdrant."
                ),
                documents=[doc_result],
            )

        # Preserve existing metadata, merge with new
        existing_meta = doc_result.metadata.model_dump()
        if new_metadata:
            existing_meta.update(new_metadata)

        now = datetime.now(timezone.utc).isoformat()
        existing_meta["updated_at"] = now
        existing_meta["relevance_score"] = doc_result.metadata.relevance_score + 2  # update = +2
        existing_meta["last_accessed_at"] = now

        # Delete old chunks
        await self.delete_all_chunks(collection_name, doc_id)

        # Generate new abstract
        abstract = None
        if self._summary_provider and self._summary_provider.enabled:
            abstract = await self._summary_provider.generate_abstract(new_content, doc_result.title)

        # Re-chunk and re-embed
        chunks = chunk_text(new_content, self._chunking_settings)
        if not chunks:
            return OperationResult(success=False, message="New content is empty.", count=0)

        embeddings = await self._embedding_provider.embed_documents(chunks)
        vector_name = self._embedding_provider.get_vector_name()

        doc_meta = DocumentMetadata(**existing_meta)

        points = []
        for i, (chunk_text_content, embedding) in enumerate(zip(chunks, embeddings)):
            payload: dict[str, Any] = {
                "document": chunk_text_content,
                "document_id": doc_id,
                "title": doc_result.title,
                "chunk_index": i,
                METADATA_PATH: doc_meta.model_dump(),
            }
            if abstract is not None:
                payload["abstract"] = abstract
            if i == 0 and doc_meta.source_type == SourceType.COMPOSED:
                payload["full_content"] = new_content

            points.append(
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embedding},
                    payload=payload,
                )
            )

        await self._client.upsert(collection_name=collection_name, points=points)

        return OperationResult(
            success=True,
            message=f"Updated document '{doc_result.title}' with {len(chunks)} chunk(s).",
            count=1,
            document_id=doc_id,
        )

    # ----- append --------------------------------------------------------

    async def append(
        self,
        filter_dict: dict,
        additional_content: str,
        *,
        collection_name: str | None = None,
    ) -> OperationResult:
        """
        Append text to an existing composed document. Fetches full_content
        from chunk 0, combines with new text, then re-chunks.

        :param filter_dict: Filter to identify the document.
        :param additional_content: Text to append.
        :param collection_name: Collection.
        :return: OperationResult.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return OperationResult(success=False, message="Collection does not exist.", count=0)

        doc_map = await self.resolve_documents(filter_dict, collection_name)

        if not doc_map:
            return OperationResult(success=False, message="No matching documents found.", count=0)

        if len(doc_map) > 1:
            docs = list(doc_map.values())
            return OperationResult(
                success=False,
                message=f"Multiple documents matched ({len(docs)}). Please specify which one by document_id.",
                documents=docs,
            )

        doc_id, doc_result = next(iter(doc_map.items()))

        if doc_result.metadata.source_type != SourceType.COMPOSED:
            return OperationResult(
                success=False,
                message=(
                    f"Cannot append to '{doc_result.title}' — it is a '{doc_result.metadata.source_type}' entry "
                    f"(source: {doc_result.metadata.source_ref}). "
                    f"No full_content available. Fetch the full text from the source and use qdrant-update instead."
                ),
                documents=[doc_result],
            )

        # Get full_content from chunk_index 0
        full_content = await self._get_full_content(collection_name, doc_id)
        if full_content is None:
            return OperationResult(
                success=False,
                message=f"Could not retrieve full_content for document '{doc_result.title}'.",
            )

        # Combine and re-store via update
        combined_content = full_content + "\n\n" + additional_content
        return await self.update(
            {"document_id": doc_id},
            combined_content,
            collection_name=collection_name,
        )

    # ----- set_metadata --------------------------------------------------

    async def set_metadata(
        self,
        filter_dict: dict,
        metadata: Metadata,
        *,
        collection_name: str | None = None,
    ) -> OperationResult:
        """
        Update metadata on all chunks of matching documents.
        Allowed for ALL source types.

        :param filter_dict: Filter to identify documents.
        :param metadata: Metadata fields to set/update.
        :param collection_name: Collection.
        :return: OperationResult.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return OperationResult(success=False, message="Collection does not exist.", count=0)

        doc_map = await self.resolve_documents(filter_dict, collection_name)
        if not doc_map:
            return OperationResult(success=False, message="No matching documents found.", count=0)

        now = datetime.now(timezone.utc).isoformat()
        metadata["updated_at"] = now
        updated_count = 0

        for doc_id, doc_result in doc_map.items():
            # Get all chunks for this document
            chunks = await self._get_all_chunks(collection_name, doc_id)
            for chunk in chunks:
                existing_meta = chunk.payload.get(METADATA_PATH, {})
                updated_meta = existing_meta.copy()
                updated_meta.update(metadata)

                await self._client.set_payload(
                    collection_name=collection_name,
                    payload={METADATA_PATH: updated_meta},
                    points=[chunk.id],
                )
            updated_count += 1

        return OperationResult(
            success=True,
            message=f"Updated metadata on {updated_count} document(s).",
            count=updated_count,
        )

    # ----- add_tags ------------------------------------------------------

    async def add_tags(
        self,
        filter_dict: dict,
        tags: list[str],
        *,
        collection_name: str | None = None,
    ) -> OperationResult:
        """
        Add tags to all chunks of matching documents.
        Allowed for ALL source types.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return OperationResult(success=False, message="Collection does not exist.", count=0)

        doc_map = await self.resolve_documents(filter_dict, collection_name)
        if not doc_map:
            return OperationResult(success=False, message="No matching documents found.", count=0)

        now = datetime.now(timezone.utc).isoformat()
        updated_count = 0

        for doc_id in doc_map:
            chunks = await self._get_all_chunks(collection_name, doc_id)
            for chunk in chunks:
                existing_meta = chunk.payload.get(METADATA_PATH, {})
                existing_tags = existing_meta.get("tags", [])
                merged_tags = list(set(existing_tags + tags))

                updated_meta = existing_meta.copy()
                updated_meta["tags"] = merged_tags
                updated_meta["updated_at"] = now

                await self._client.set_payload(
                    collection_name=collection_name,
                    payload={METADATA_PATH: updated_meta},
                    points=[chunk.id],
                )
            updated_count += 1

        return OperationResult(
            success=True,
            message=f"Added tags {tags} to {updated_count} document(s).",
            count=updated_count,
        )

    # ----- remove_tags ---------------------------------------------------

    async def remove_tags(
        self,
        filter_dict: dict,
        tags: list[str],
        *,
        collection_name: str | None = None,
    ) -> OperationResult:
        """
        Remove tags from all chunks of matching documents.
        Allowed for ALL source types.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return OperationResult(success=False, message="Collection does not exist.", count=0)

        doc_map = await self.resolve_documents(filter_dict, collection_name)
        if not doc_map:
            return OperationResult(success=False, message="No matching documents found.", count=0)

        now = datetime.now(timezone.utc).isoformat()
        updated_count = 0

        for doc_id in doc_map:
            chunks = await self._get_all_chunks(collection_name, doc_id)
            for chunk in chunks:
                existing_meta = chunk.payload.get(METADATA_PATH, {})
                existing_tags = existing_meta.get("tags", [])
                filtered_tags = [t for t in existing_tags if t not in tags]

                updated_meta = existing_meta.copy()
                updated_meta["tags"] = filtered_tags
                updated_meta["updated_at"] = now

                await self._client.set_payload(
                    collection_name=collection_name,
                    payload={METADATA_PATH: updated_meta},
                    points=[chunk.id],
                )
            updated_count += 1

        return OperationResult(
            success=True,
            message=f"Removed tags {tags} from {updated_count} document(s).",
            count=updated_count,
        )

    # ----- resolve / chunk management ------------------------------------

    async def resolve_documents(
        self,
        filter_dict: dict,
        collection_name: str,
    ) -> dict[str, DocumentResult]:
        """
        Find all documents matching a filter. Returns a dict of
        document_id → DocumentResult.
        """
        qdrant_filter = self._build_filter(filter_dict)
        if not qdrant_filter:
            return {}

        results, _ = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            return {}

        doc_map: dict[str, DocumentResult] = {}
        for point in results:
            payload = point.payload
            doc_id = payload.get("document_id")
            if not doc_id:
                continue
            if doc_id not in doc_map:
                meta_dict = payload.get(METADATA_PATH, {})
                doc_map[doc_id] = DocumentResult(
                    document_id=doc_id,
                    title=payload.get("title", "(untitled)"),
                    abstract=payload.get("abstract"),
                    metadata=DocumentMetadata(**meta_dict),
                    chunk_count=1,
                )
            else:
                doc_map[doc_id].chunk_count += 1

        return doc_map

    async def delete_all_chunks(self, collection_name: str, document_id: str):
        """Delete all points belonging to a document_id."""
        doc_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )
        await self._client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=doc_filter),
        )

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _build_filter(self, filter_dict: dict) -> models.Filter | None:
        """
        Convert a simple dict into a Qdrant Filter.

        Supports:
        - Top-level fields: document_id, title
        - Content field: content/document → text match on 'document'
        - Metadata fields: everything else → metadata.{key}
        """
        if not filter_dict:
            return None

        conditions = []
        for key, value in filter_dict.items():
            # Top-level exact-match fields
            if key == "document_id":
                conditions.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=value),
                    )
                )
            elif key == "title":
                conditions.append(
                    models.FieldCondition(
                        key="title",
                        match=models.MatchText(text=value),
                    )
                )
            elif key in ("content", "document"):
                if isinstance(value, str):
                    conditions.append(
                        models.FieldCondition(
                            key="document",
                            match=models.MatchText(text=value),
                        )
                    )
            else:
                # Metadata fields
                field_key = (
                    key
                    if key.startswith(f"{METADATA_PATH}.")
                    else f"{METADATA_PATH}.{key}"
                )
                if isinstance(value, list):
                    conditions.append(
                        models.FieldCondition(
                            key=field_key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=field_key,
                            match=models.MatchValue(value=value),
                        )
                    )

        return models.Filter(must=conditions) if conditions else None

    def _group_points_to_documents(self, points: list) -> list[DocumentResult]:
        """
        Group a list of Qdrant points (from search or scroll) into
        deduplicated DocumentResults, ordered by first appearance.
        """
        doc_map: dict[str, DocumentResult] = {}
        order: list[str] = []

        for point in points:
            payload = point.payload
            doc_id = payload.get("document_id")
            if not doc_id:
                # Legacy point without document_id — wrap as single document
                doc_id = f"legacy_{point.id}"

            if doc_id not in doc_map:
                meta_dict = payload.get(METADATA_PATH, {})
                doc_map[doc_id] = DocumentResult(
                    document_id=doc_id,
                    title=payload.get("title", "(untitled)"),
                    abstract=payload.get("abstract"),
                    metadata=DocumentMetadata(**meta_dict),
                    chunk_count=1,
                )
                order.append(doc_id)
            else:
                doc_map[doc_id].chunk_count += 1

        return [doc_map[doc_id] for doc_id in order]

    async def _get_all_chunks(self, collection_name: str, document_id: str) -> list:
        """Get all Qdrant points belonging to a document_id."""
        doc_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )
        results, _ = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=doc_filter,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
        return results

    async def _get_full_content(self, collection_name: str, document_id: str) -> str | None:
        """Retrieve full_content from chunk_index 0 of a composed document."""
        doc_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
                models.FieldCondition(
                    key="chunk_index",
                    match=models.MatchValue(value=0),
                ),
            ]
        )
        results, _ = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=doc_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if results:
            return results[0].payload.get("full_content")
        return None

    async def _update_access_tracking_by_document_ids(
        self,
        collection_name: str,
        document_ids: list[str],
        score_increment: int,
    ):
        """
        Update access tracking on ALL chunks of the given document_ids.
        """
        if not document_ids:
            return

        now = datetime.now(timezone.utc).isoformat()

        for doc_id in document_ids:
            chunks = await self._get_all_chunks(collection_name, doc_id)
            for point in chunks:
                existing_meta = point.payload.get(METADATA_PATH, {})
                current_score = existing_meta.get("relevance_score", 0)

                updated_meta = existing_meta.copy()
                updated_meta["relevance_score"] = current_score + score_increment
                updated_meta["last_accessed_at"] = now

                await self._client.set_payload(
                    collection_name=collection_name,
                    payload={METADATA_PATH: updated_meta},
                    points=[point.id],
                )

    async def _count_by_filter(
        self,
        collection_name: str,
        qdrant_filter: models.Filter,
    ) -> int:
        """Count entries matching a filter."""
        result = await self._client.count(
            collection_name=collection_name,
            count_filter=qdrant_filter,
            exact=True,
        )
        return result.count

    async def _ensure_collection_exists(self, collection_name: str):
        """Ensure the collection exists, creating it with indexes if necessary."""
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            vector_size = self._embedding_provider.get_vector_size()
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

            # Create payload indexes for the two-level model
            default_indexes: dict[str, models.PayloadSchemaType] = {
                "document_id": models.PayloadSchemaType.KEYWORD,
                "chunk_index": models.PayloadSchemaType.INTEGER,
                f"{METADATA_PATH}.source_type": models.PayloadSchemaType.KEYWORD,
                f"{METADATA_PATH}.category": models.PayloadSchemaType.KEYWORD,
                f"{METADATA_PATH}.tags": models.PayloadSchemaType.KEYWORD,
            }

            # Merge with any user-configured indexes
            if self._field_indexes:
                default_indexes.update(self._field_indexes)

            for field_name, field_type in default_indexes.items():
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
