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
        :param new_metadata: New metadata (optional).
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

        # Find existing entries to preserve created_at
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

        # Prepare metadata with updated_at timestamp
        metadata = new_metadata or {}
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        # Preserve created_at from first existing entry if not in new metadata
        if "created_at" not in metadata and existing_entries:
            old_metadata = existing_entries[0].metadata or {}
            if "created_at" in old_metadata:
                metadata["created_at"] = old_metadata["created_at"]

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

        # Count entries before update (for return value)
        count = await self._count_by_filter(collection_name, qdrant_filter)
        if count == 0:
            return 0

        # Add updated_at timestamp
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Build payload with metadata path prefix
        payload_updates = {f"{METADATA_PATH}.{key}": value for key, value in metadata.items()}

        # Update metadata
        await self._client.set_payload(
            collection_name=collection_name,
            payload=payload_updates,
            points=models.FilterSelector(filter=qdrant_filter),
        )

        return count

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