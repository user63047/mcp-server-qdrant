from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    ChunkingSettings,
    EmbeddingProviderSettings,
    QdrantSettings,
    SummarySettings,
    ToolSettings,
)

mcp = QdrantMCPServer(
    tool_settings=ToolSettings(),
    qdrant_settings=QdrantSettings(),
    chunking_settings=ChunkingSettings(),
    summary_settings=SummarySettings(),
    embedding_provider_settings=EmbeddingProviderSettings(),
)
