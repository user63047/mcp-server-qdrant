from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Store a document in Qdrant. Provide a title and the text content. "
    "The text is automatically split into chunks for embedding. "
    "Optionally provide metadata: category (str), tags (list[str]), "
    "source_type ('composed' by default, or 'trilium', 'pdf', 'paperless'), "
    "source_ref (URL/path to external source)."
)
DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Search for documents in Qdrant by semantic similarity. Returns document-level results "
    "with title, abstract, and metadata. Use this tool when you need to: \n"
    " - Find documents by their content \n"
    " - Access knowledge for further analysis \n"
    " - Get personal information about the user \n"
    "Multiple chunks of the same document are grouped automatically."
)
DEFAULT_TOOL_DELETE_DESCRIPTION = (
    "Delete a document and all its chunks from Qdrant. "
    "Only works for 'composed' entries (source_type='composed'). "
    "For external sources (trilium, pdf, etc.), delete at the source — "
    "the sync will update Qdrant automatically. "
    "If multiple documents match the filter, they are returned for disambiguation — "
    "use document_id for exact targeting. "
    "Filter by 'document_id', 'category', 'tags', or 'content'."
)
DEFAULT_TOOL_UPDATE_DESCRIPTION = (
    "Replace the entire content of a document in Qdrant (re-chunks and creates new embeddings). "
    "Only works for 'composed' entries. For external sources, edit the source directly. "
    "If you want to APPEND text, use qdrant-append instead. "
    "If multiple documents match, they are returned for disambiguation — use document_id for exact targeting. "
    "Filter by 'document_id', 'category', 'tags', or 'content'."
)
DEFAULT_TOOL_SET_METADATA_DESCRIPTION = (
    "Update metadata on documents without changing content or embeddings. "
    "Works for ALL source types (composed, trilium, pdf, etc.). "
    "Updates are applied to all chunks of the matching document(s). "
    "Filter by 'document_id', 'category', 'tags', or 'content'."
)
DEFAULT_TOOL_LIST_DESCRIPTION = (
    "List documents in a Qdrant collection, grouped by document. "
    "Returns title, abstract, and metadata per document. "
    "Optionally filter by metadata fields like 'category', 'tags', 'source_type'."
)
DEFAULT_TOOL_COLLECTIONS_DESCRIPTION = (
    "List all available collections in Qdrant."
)
DEFAULT_TOOL_ADD_TAGS_DESCRIPTION = (
    "Add tags to documents without removing existing tags. "
    "Works for ALL source types. Updates all chunks of the matching document(s). "
    "Filter by 'document_id', 'category', 'tags', or 'content'."
)
DEFAULT_TOOL_REMOVE_TAGS_DESCRIPTION = (
    "Remove specific tags from documents. "
    "Works for ALL source types. Updates all chunks of the matching document(s). "
    "Filter by 'document_id', 'category', 'tags', or 'content'."
)
DEFAULT_TOOL_APPEND_DESCRIPTION = (
    "Append text to an existing document in Qdrant. The existing and new text are combined, "
    "then the document is re-chunked and re-embedded. "
    "Only works for 'composed' entries (full_content is stored in Qdrant). "
    "For external sources (trilium, pdf, etc.), the full content is not available — "
    "fetch it from the source via the appropriate MCP server, combine with the new text, "
    "and use qdrant-update instead. "
    "If the LLM has enriched content originally from an external source, "
    "ask the user whether the enriched version should be written back to the source. "
    "If multiple documents match, they are returned for disambiguation — use document_id for exact targeting. "
    "Filter by 'document_id', 'category', 'tags', or 'content'."
)

METADATA_PATH = "metadata"


class ToolSettings(BaseSettings):
    """
    Configuration for all the tools.
    """

    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION",
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION",
    )
    tool_delete_description: str = Field(
        default=DEFAULT_TOOL_DELETE_DESCRIPTION,
        validation_alias="TOOL_DELETE_DESCRIPTION",
    )
    tool_update_description: str = Field(
        default=DEFAULT_TOOL_UPDATE_DESCRIPTION,
        validation_alias="TOOL_UPDATE_DESCRIPTION",
    )
    tool_set_metadata_description: str = Field(
        default=DEFAULT_TOOL_SET_METADATA_DESCRIPTION,
        validation_alias="TOOL_SET_METADATA_DESCRIPTION",
    )
    tool_list_description: str = Field(
        default=DEFAULT_TOOL_LIST_DESCRIPTION,
        validation_alias="TOOL_LIST_DESCRIPTION",
    )
    tool_collections_description: str = Field(
        default=DEFAULT_TOOL_COLLECTIONS_DESCRIPTION,
        validation_alias="TOOL_COLLECTIONS_DESCRIPTION",
    )
    tool_add_tags_description: str = Field(
        default=DEFAULT_TOOL_ADD_TAGS_DESCRIPTION,
        validation_alias="TOOL_ADD_TAGS_DESCRIPTION",
    )
    tool_remove_tags_description: str = Field(
        default=DEFAULT_TOOL_REMOVE_TAGS_DESCRIPTION,
        validation_alias="TOOL_REMOVE_TAGS_DESCRIPTION",
    )
    tool_append_description: str = Field(
        default=DEFAULT_TOOL_APPEND_DESCRIPTION,
        validation_alias="TOOL_APPEND_DESCRIPTION",
    )


class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_URL",
    )


class FilterableField(BaseModel):
    name: str = Field(description="The name of the field payload field to filter on")
    description: str = Field(
        description="A description for the field used in the tool description"
    )
    field_type: Literal["keyword", "integer", "float", "boolean"] = Field(
        description="The type of the field"
    )
    condition: Literal["==", "!=", ">", ">=", "<", "<=", "any", "except"] | None = (
        Field(
            default=None,
            description=(
                "The condition to use for the filter. If not provided, the field will be indexed, but no "
                "filter argument will be exposed to MCP tool."
            ),
        )
    )
    required: bool = Field(
        default=False,
        description="Whether the field is required for the filter.",
    )


class ChunkingSettings(BaseSettings):
    """
    Configuration for the chunking strategy.
    """

    chunk_size: int = Field(
        default=1500,
        validation_alias="CHUNK_SIZE",
        description="Target chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=375,
        validation_alias="CHUNK_OVERLAP",
        description="Overlap between chunks in tokens",
    )


class SummarySettings(BaseSettings):
    """
    Configuration for the summary/abstract generation.
    """

    summary_model: str | None = Field(
        default=None,
        validation_alias="SUMMARY_MODEL",
        description="Ollama model for generating abstracts (e.g. 'gemma3:4b'). None disables summaries.",
    )
    summary_provider: str = Field(
        default="ollama",
        validation_alias="SUMMARY_PROVIDER",
        description="Summary provider (currently only 'ollama')",
    )


class QdrantSettings(BaseSettings):
    """
    Configuration for the Qdrant connector.
    """

    location: str | None = Field(default=None, validation_alias="QDRANT_URL")
    api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str | None = Field(
        default=None, validation_alias="COLLECTION_NAME"
    )
    local_path: str | None = Field(default=None, validation_alias="QDRANT_LOCAL_PATH")
    search_limit: int = Field(default=10, validation_alias="QDRANT_SEARCH_LIMIT")
    read_only: bool = Field(default=False, validation_alias="QDRANT_READ_ONLY")

    filterable_fields: list[FilterableField] | None = Field(default=None)

    allow_arbitrary_filter: bool = Field(
        default=False, validation_alias="QDRANT_ALLOW_ARBITRARY_FILTER"
    )

    def filterable_fields_dict(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {field.name: field for field in self.filterable_fields}

    def filterable_fields_dict_with_conditions(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {
            field.name: field
            for field in self.filterable_fields
            if field.condition is not None
        }

    @model_validator(mode="after")
    def check_local_path_conflict(self) -> "QdrantSettings":
        if self.local_path:
            if self.location is not None or self.api_key is not None:
                raise ValueError(
                    "If 'local_path' is set, 'location' and 'api_key' must be None."
                )
        return self