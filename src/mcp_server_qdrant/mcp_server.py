import json
import logging
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        # ===== STORE TOOL =====
        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        # ===== FIND TOOL =====
        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            """
            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        # ===== DELETE TOOL =====
        async def delete(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to delete from")
            ],
            filter: Annotated[
                dict, Field(description="Filter criteria (e.g. {'category': 'test'})")
            ],
        ) -> str:
            """
            Delete entries matching the filter from Qdrant.
            """
            await ctx.debug(f"Deleting entries with filter {filter} from {collection_name}")

            count = await self.qdrant_connector.delete(
                filter, collection_name=collection_name
            )
            return f"Deleted {count} entries from {collection_name}"

        # ===== UPDATE TOOL =====
        async def update(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to update")
            ],
            filter: Annotated[
                dict, Field(description="Filter to identify entries to update")
            ],
            new_information: Annotated[
                str, Field(description="New content for the entries")
            ],
            new_metadata: Annotated[
                Metadata | None, Field(description="New metadata (optional)")
            ] = None,
        ) -> str:
            """
            Update entries matching the filter with new content. This creates new embeddings.
            """
            await ctx.debug(f"Updating entries with filter {filter} in {collection_name}")

            count = await self.qdrant_connector.update(
                filter, new_information, new_metadata, collection_name=collection_name
            )
            return f"Updated {count} entries in {collection_name}"

        # ===== SET METADATA TOOL =====
        async def set_metadata(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection")
            ],
            filter: Annotated[
                dict, Field(description="Filter to identify entries")
            ],
            metadata: Annotated[
                dict, Field(description="Metadata fields to set/update")
            ],
        ) -> str:
            """
            Set or update metadata on entries matching the filter without changing embeddings.
            """
            await ctx.debug(f"Setting metadata on entries with filter {filter} in {collection_name}")

            count = await self.qdrant_connector.set_metadata(
                filter, metadata, collection_name=collection_name
            )
            return f"Updated metadata on {count} entries in {collection_name}"

        # ===== LIST TOOL =====
        async def list_entries(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to list")
            ],
            filter: Annotated[
                dict | None, Field(description="Optional filter criteria")
            ] = None,
            limit: Annotated[
                int, Field(description="Maximum number of entries to return")
            ] = 10,
        ) -> list[str] | None:
            """
            List entries in a collection, optionally filtered by metadata.
            """
            await ctx.debug(f"Listing entries from {collection_name} with filter {filter}")

            entries = await self.qdrant_connector.list_entries(
                filter, limit=limit, collection_name=collection_name
            )
            if not entries:
                return None
            content = [
                f"Entries in collection '{collection_name}' (limit: {limit})",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        # ===== COLLECTIONS TOOL =====
        async def collections(ctx: Context) -> list[str]:
            """
            List all available collections in Qdrant.
            """
            await ctx.debug("Listing all collections")

            names = await self.qdrant_connector.get_collection_names()
            if not names:
                return ["No collections found"]
            return [f"Available collections: {', '.join(names)}"]

        # ===== TOOL REGISTRATION =====
        
        # Prepare function references
        find_foo = find
        store_foo = store
        delete_foo = delete
        update_foo = update
        set_metadata_foo = set_metadata
        list_entries_foo = list_entries
        collections_foo = collections

        # Handle filterable conditions for find
        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})

        # Handle default collection_name
        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            delete_foo = make_partial_function(
                delete_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            update_foo = make_partial_function(
                update_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            set_metadata_foo = make_partial_function(
                set_metadata_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            list_entries_foo = make_partial_function(
                list_entries_foo, {"collection_name": self.qdrant_settings.collection_name}
            )

        # Register read-only tools (always available)
        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )
        self.tool(
            list_entries_foo,
            name="qdrant-list",
            description=self.tool_settings.tool_list_description,
        )
        self.tool(
            collections_foo,
            name="qdrant-collections",
            description=self.tool_settings.tool_collections_description,
        )

        # Register write tools (only if not read_only)
        if not self.qdrant_settings.read_only:
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )
            self.tool(
                delete_foo,
                name="qdrant-delete",
                description=self.tool_settings.tool_delete_description,
            )
            self.tool(
                update_foo,
                name="qdrant-update",
                description=self.tool_settings.tool_update_description,
            )
            self.tool(
                set_metadata_foo,
                name="qdrant-set-metadata",
                description=self.tool_settings.tool_set_metadata_description,
            )
