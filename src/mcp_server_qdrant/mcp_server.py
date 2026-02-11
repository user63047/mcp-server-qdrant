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
from mcp_server_qdrant.models import DocumentResult
from mcp_server_qdrant.qdrant import ArbitraryFilter, Metadata, OperationResult, QdrantConnector
from mcp_server_qdrant.settings import (
    ChunkingSettings,
    EmbeddingProviderSettings,
    QdrantSettings,
    SummarySettings,
    ToolSettings,
)
from mcp_server_qdrant.summary import SummaryProvider, create_summary_provider

logger = logging.getLogger(__name__)


class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant using the two-level document/chunk architecture.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        chunking_settings: ChunkingSettings | None = None,
        summary_settings: SummarySettings | None = None,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.chunking_settings = chunking_settings or ChunkingSettings()
        self.summary_settings = summary_settings or SummarySettings()

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

        # Create summary provider
        self.summary_provider: SummaryProvider | None = None
        if self.summary_settings.summary_model:
            self.summary_provider = create_summary_provider(
                self.summary_settings, self.embedding_provider_settings
            )

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
            chunking_settings=self.chunking_settings,
            summary_provider=self.summary_provider,
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def _format_document(self, doc: DocumentResult) -> str:
        """Format a DocumentResult for LLM consumption."""
        return doc.format_for_llm()

    def _format_operation_result(self, result: OperationResult) -> str:
        """Format an OperationResult for LLM consumption."""
        parts = [result.message]
        if result.documents:
            parts.append("\nMatching documents:")
            for doc in result.documents:
                parts.append(doc.format_for_llm())
        return "\n".join(parts)

    def setup_tools(self):
        """Register the tools in the server."""

        # ===== STORE TOOL =====
        async def store(
            ctx: Context,
            title: Annotated[str, Field(description="Title of the document to store")],
            information: Annotated[str, Field(description="Text content to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the document in")
            ],
            metadata: Annotated[
                Metadata | None,
                Field(
                    description=(
                        "Optional metadata: category (str), tags (list[str]), "
                        "source_type ('composed'|'trilium'|'pdf'|'paperless'), "
                        "source_ref (url/path to external source). "
                        "Default source_type is 'composed'."
                    )
                ),
            ] = None,
        ) -> str:
            """
            Store a document in Qdrant. The text is automatically chunked if it
            exceeds the embedding model's context window. An abstract is generated
            if a summary model is configured.
            """
            await ctx.debug(f"Storing document '{title}' in Qdrant")

            result = await self.qdrant_connector.store(
                title, information, metadata, collection_name=collection_name
            )
            return self._format_operation_result(result)

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
            Find documents in Qdrant by semantic search. Returns document-level
            results (title, abstract, metadata). Multiple chunks of the same
            document are grouped automatically.
            """
            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query '{query}'")

            documents = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not documents:
                return None
            content = [f"Results for the query '{query}'"]
            for doc in documents:
                content.append(self._format_document(doc))
            return content

        # ===== DELETE TOOL =====
        async def delete(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to delete from")
            ],
            filter: Annotated[
                dict,
                Field(
                    description=(
                        "Filter to identify the document. Use 'document_id' for exact targeting, "
                        "or metadata fields like 'category', 'tags'. "
                        "Only 'composed' entries can be deleted; external sources must be deleted at the source."
                    )
                ),
            ],
        ) -> str:
            """
            Delete a document and all its chunks from Qdrant.
            Only works for 'composed' entries. For external sources (trilium, pdf, etc.),
            delete at the source and the sync will update Qdrant.
            If multiple documents match, returns them for disambiguation.
            """
            await ctx.debug(f"Deleting document with filter {filter} from {collection_name}")

            result = await self.qdrant_connector.delete(
                filter, collection_name=collection_name
            )
            return self._format_operation_result(result)

        # ===== UPDATE TOOL =====
        async def update(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to update")
            ],
            filter: Annotated[
                dict,
                Field(
                    description=(
                        "Filter to identify the document. Use 'document_id' for exact targeting, "
                        "or metadata fields like 'category', 'tags', or 'content' for text search."
                    )
                ),
            ],
            new_information: Annotated[
                str, Field(description="New content that REPLACES the entire document")
            ],
            new_metadata: Annotated[
                Metadata | None, Field(description="New metadata to merge with existing (optional)")
            ] = None,
        ) -> str:
            """
            Replace the content of an existing document. Creates new chunks and embeddings.
            Only works for 'composed' entries. For external sources, edit the source directly.
            If multiple documents match, returns them for disambiguation.
            """
            await ctx.debug(f"Updating document with filter {filter} in {collection_name}")

            result = await self.qdrant_connector.update(
                filter, new_information, new_metadata, collection_name=collection_name
            )
            return self._format_operation_result(result)

        # ===== APPEND TOOL =====
        async def append(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection")
            ],
            filter: Annotated[
                dict,
                Field(
                    description=(
                        "Filter to identify the document. Use 'document_id' for exact targeting, "
                        "or metadata fields like 'category', 'tags', or 'content' for text search."
                    )
                ),
            ],
            additional_text: Annotated[
                str, Field(description="Text to append to the existing document")
            ],
        ) -> str:
            """
            Append text to an existing document. The existing and new text are combined,
            then the document is re-chunked and re-embedded.
            Only works for 'composed' entries. For external sources, the full content
            is not stored in Qdrant — fetch it from the source, combine, and use qdrant-update.
            If multiple documents match, returns them for disambiguation.
            """
            await ctx.debug(f"Appending to document with filter {filter} in {collection_name}")

            result = await self.qdrant_connector.append(
                filter, additional_text, collection_name=collection_name
            )
            return self._format_operation_result(result)

        # ===== SET METADATA TOOL =====
        async def set_metadata(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection")
            ],
            filter: Annotated[
                dict, Field(description="Filter to identify documents")
            ],
            metadata: Annotated[
                dict, Field(description="Metadata fields to set/update (e.g. category, tags)")
            ],
        ) -> str:
            """
            Set or update metadata on documents matching the filter without changing
            content or embeddings. Works for ALL source types.
            """
            await ctx.debug(f"Setting metadata on documents with filter {filter} in {collection_name}")

            result = await self.qdrant_connector.set_metadata(
                filter, metadata, collection_name=collection_name
            )
            return self._format_operation_result(result)

        # ===== ADD TAGS TOOL =====
        async def add_tags(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection")
            ],
            filter: Annotated[
                dict, Field(description="Filter to identify documents")
            ],
            tags: Annotated[
                list[str], Field(description="Tags to add")
            ],
        ) -> str:
            """
            Add tags to documents matching the filter without removing existing tags.
            Works for ALL source types.
            """
            await ctx.debug(f"Adding tags {tags} to documents with filter {filter} in {collection_name}")

            result = await self.qdrant_connector.add_tags(
                filter, tags, collection_name=collection_name
            )
            return self._format_operation_result(result)

        # ===== REMOVE TAGS TOOL =====
        async def remove_tags(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection")
            ],
            filter: Annotated[
                dict, Field(description="Filter to identify documents")
            ],
            tags: Annotated[
                list[str], Field(description="Tags to remove")
            ],
        ) -> str:
            """
            Remove specific tags from documents matching the filter.
            Works for ALL source types.
            """
            await ctx.debug(f"Removing tags {tags} from documents with filter {filter} in {collection_name}")

            result = await self.qdrant_connector.remove_tags(
                filter, tags, collection_name=collection_name
            )
            return self._format_operation_result(result)

        # ===== LIST TOOL =====
        async def list_entries(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to list")
            ],
            filter: Annotated[
                dict | None, Field(description="Optional filter criteria (category, tags, source_type, ...)")
            ] = None,
            limit: Annotated[
                int, Field(description="Maximum number of documents to return")
            ] = 10,
        ) -> list[str] | None:
            """
            List documents in a collection, grouped by document. Optionally filtered
            by metadata fields. Returns title, abstract, and metadata per document.
            """
            await ctx.debug(f"Listing documents from {collection_name} with filter {filter}")

            documents = await self.qdrant_connector.list_entries(
                filter, limit=limit, collection_name=collection_name
            )
            if not documents:
                return None
            content = [f"Documents in collection '{collection_name}' (limit: {limit})"]
            for doc in documents:
                content.append(self._format_document(doc))
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
        append_foo = append
        set_metadata_foo = set_metadata
        list_entries_foo = list_entries
        collections_foo = collections
        add_tags_foo = add_tags
        remove_tags_foo = remove_tags

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
            append_foo = make_partial_function(
                append_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            set_metadata_foo = make_partial_function(
                set_metadata_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            list_entries_foo = make_partial_function(
                list_entries_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            add_tags_foo = make_partial_function(
                add_tags_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            remove_tags_foo = make_partial_function(
                remove_tags_foo, {"collection_name": self.qdrant_settings.collection_name}
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
                append_foo,
                name="qdrant-append",
                description=self.tool_settings.tool_append_description,
            )
            self.tool(
                set_metadata_foo,
                name="qdrant-set-metadata",
                description=self.tool_settings.tool_set_metadata_description,
            )
            self.tool(
                add_tags_foo,
                name="qdrant-add-tags",
                description=self.tool_settings.tool_add_tags_description,
            )
            self.tool(
                remove_tags_foo,
                name="qdrant-remove-tags",
                description=self.tool_settings.tool_remove_tags_description,
            )
