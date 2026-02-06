import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None


class EntryWithId(BaseModel):
    """
    A single entry in the Qdrant collection with its point ID.
    """

    id: str
    content: str
    metadata: Metadata | None = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add created_at timestamp if not present
        metadata = entry.metadata or {}
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(timezone.utc).isoformat()

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, METADATA_PATH: metadata}
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
            )
            for result in search_results.points
        ]

    async def delete(
        self,
        filter_dict: dict,
        *,
        collection_name: str | None = None,
    ) -> int:
        """
        Delete entries based on a filter.
        :param filter_dict: The filter criteria (e.g. {"category": "test"}).
        :param collection_name: The name of the collection to delete from.
        :return: Number of deleted entries (estimated).
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return 0

        qdrant_filter = self._build_filter(filter_dict)
        if not qdrant_filter:
            return 0

        # Count entries before deletion (for return value)
        count_before = await self._count_by_filter(collection_name, qdrant_filter)

        # Delete entries matching the filter
        await self._client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=qdrant_filter),
        )

        return count_before

    async def update(
        self,
        filter_dict: dict,
        new_content: str,
        new_metadata: Metadata | None = None,
        *,
        collection_name: str | None = None,
    ) -> int:
        """
        Update entries matching the filter with new content (creates new embeddings).
        :param filter_dict: The filter criteria to identify entries.
        :param new_content: The new text content.
        :param new_metadata: New metadata to merge with existing (optional).
        :param collection_name: The name of the collection.
        :return: Number of updated entries.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return 0

        qdrant_filter = self._build_filter(filter_dict)
        if not qdrant_filter:
            return 0

        # Find existing entries to preserve metadata
        existing_entries = await self._scroll_entries(collection_name, qdrant_filter, limit=100)
        if not existing_entries:
            return 0

        # Delete old entries
        await self._client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=qdrant_filter),
        )

        # Create new embedding
        embeddings = await self._embedding_provider.embed_documents([new_content])
        vector_name = self._embedding_provider.get_vector_name()

        # Preserve old metadata and merge with new
        old_metadata = existing_entries[0].metadata or {}
        metadata = old_metadata.copy()
        
        # Merge new metadata if provided (overwrites existing keys)
        if new_metadata:
            metadata.update(new_metadata)
        
        # Always update the updated_at timestamp
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Store new entry
        payload = {"document": new_content, METADATA_PATH: metadata}
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )

        return len(existing_entries)

    async def set_metadata(
        self,
        filter_dict: dict,
        metadata: Metadata,
        *,
        collection_name: str | None = None,
    ) -> int:
        """
        Set/update metadata on entries matching the filter without creating new embeddings.
        :param filter_dict: The filter criteria to identify entries.
        :param metadata: The metadata fields to set/update.
        :param collection_name: The name of the collection.
        :return: Number of updated entries.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return 0

        qdrant_filter = self._build_filter(filter_dict)
        if not qdrant_filter:
            return 0

        # Get entries with IDs so we can update them properly
        entries_with_ids = await self._scroll_entries_with_ids(collection_name, qdrant_filter, limit=100)
        if not entries_with_ids:
            return 0

        # Add updated_at timestamp
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Update each entry's metadata properly
        for entry in entries_with_ids:
            # Merge existing metadata with new metadata
            existing_metadata = entry.metadata or {}
            updated_metadata = existing_metadata.copy()
            updated_metadata.update(metadata)
            
            # Overwrite the entire metadata object
            await self._client.set_payload(
                collection_name=collection_name,
                payload={METADATA_PATH: updated_metadata},
                points=[entry.id],
            )

        return len(entries_with_ids)

    async def add_tags(
        self,
        filter_dict: dict,
        tags: list[str],
        *,
        collection_name: str | None = None,
    ) -> int:
        """
        Add tags to entries matching the filter without removing existing tags.
        :param filter_dict: The filter criteria to identify entries.
        :param tags: The tags to add.
        :param collection_name: The name of the collection.
        :return: Number of updated entries.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return 0

        qdrant_filter = self._build_filter(filter_dict)
        if not qdrant_filter:
            return 0

        # Get entries with IDs so we can update them properly
        entries_with_ids = await self._scroll_entries_with_ids(collection_name, qdrant_filter, limit=100)
        if not entries_with_ids:
            return 0

        # Update each entry's tags
        for entry in entries_with_ids:
            existing_metadata = entry.metadata or {}
            existing_tags = existing_metadata.get("tags", [])
            
            # Merge tags (avoid duplicates)
            merged_tags = list(set(existing_tags + tags))
            
            # Update metadata
            updated_metadata = existing_metadata.copy()
            updated_metadata["tags"] = merged_tags
            updated_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            await self._client.set_payload(
                collection_name=collection_name,
                payload={METADATA_PATH: updated_metadata},
                points=[entry.id],
            )

        return len(entries_with_ids)

    async def remove_tags(
        self,
        filter_dict: dict,
        tags: list[str],
        *,
        collection_name: str | None = None,
    ) -> int:
        """
        Remove specific tags from entries matching the filter.
        :param filter_dict: The filter criteria to identify entries.
        :param tags: The tags to remove.
        :param collection_name: The name of the collection.
        :return: Number of updated entries.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return 0

        qdrant_filter = self._build_filter(filter_dict)
        if not qdrant_filter:
            return 0

        # Get entries with IDs so we can update them properly
        entries_with_ids = await self._scroll_entries_with_ids(collection_name, qdrant_filter, limit=100)
        if not entries_with_ids:
            return 0

        # Update each entry's tags
        for entry in entries_with_ids:
            existing_metadata = entry.metadata or {}
            existing_tags = existing_metadata.get("tags", [])
            
            # Remove specified tags
            filtered_tags = [t for t in existing_tags if t not in tags]
            
            # Update metadata
            updated_metadata = existing_metadata.copy()
            updated_metadata["tags"] = filtered_tags
            updated_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            await self._client.set_payload(
                collection_name=collection_name,
                payload={METADATA_PATH: updated_metadata},
                points=[entry.id],
            )

        return len(entries_with_ids)

    async def list_entries(
        self,
        filter_dict: dict | None = None,
        limit: int = 10,
        *,
        collection_name: str | None = None,
    ) -> list[Entry]:
        """
        List entries in a collection, optionally filtered.
        :param filter_dict: Optional filter criteria.
        :param limit: Maximum number of entries to return.
        :param collection_name: The name of the collection.
        :return: List of entries.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        qdrant_filter = self._build_filter(filter_dict) if filter_dict else None
        
        return await self._scroll_entries(collection_name, qdrant_filter, limit)

    def _build_filter(self, filter_dict: dict) -> models.Filter | None:
        """
        Convert a simple dict into a Qdrant Filter.
        
        Input:  {"category": "homelab", "source": "chat"}
        Output: models.Filter with must-conditions
        """
        if not filter_dict:
            return None
            
        conditions = []
        for key, value in filter_dict.items():
            # Add metadata prefix if not already present
            field_key = f"{METADATA_PATH}.{key}" if not key.startswith(f"{METADATA_PATH}.") else key
            
            if isinstance(value, list):
                # Match any of the values
                conditions.append(
                    models.FieldCondition(
                        key=field_key,
                        match=models.MatchAny(any=value)
                    )
                )
            else:
                # Exact match
                conditions.append(
                    models.FieldCondition(
                        key=field_key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        return models.Filter(must=conditions) if conditions else None

    async def _scroll_entries(
        self,
        collection_name: str,
        qdrant_filter: models.Filter | None,
        limit: int,
    ) -> list[Entry]:
        """
        Scroll through entries in a collection.
        """
        results, _ = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            Entry(
                content=point.payload["document"],
                metadata=point.payload.get("metadata"),
            )
            for point in results
        ]

    async def _scroll_entries_with_ids(
        self,
        collection_name: str,
        qdrant_filter: models.Filter | None,
        limit: int,
    ) -> list[EntryWithId]:
        """
        Scroll through entries in a collection, including point IDs.
        """
        results, _ = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            EntryWithId(
                id=point.id,
                content=point.payload["document"],
                metadata=point.payload.get("metadata"),
            )
            for point in results
        ]

    async def _count_by_filter(
        self,
        collection_name: str,
        qdrant_filter: models.Filter,
    ) -> int:
        """
        Count entries matching a filter.
        """
        result = await self._client.count(
            collection_name=collection_name,
            count_filter=qdrant_filter,
            exact=True,
        )
        return result.count

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
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

            # Create payload indexes if configured
            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )