FROM python:3.11-slim

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Install mcp-server-qdrant from local source
COPY . .
RUN uv pip install --system --no-cache-dir .

# Port for combined MCP + REST API
EXPOSE 8000

# Environment variables with sensible defaults for homelab deployment
# --- Embedding ---
ENV EMBEDDING_PROVIDER="ollama"
ENV EMBEDDING_MODEL="embeddinggemma:300m"
ENV OLLAMA_URL=""

# --- Summary ---
ENV SUMMARY_MODEL=""
ENV SUMMARY_PROVIDER="ollama"

# --- Qdrant ---
ENV QDRANT_URL=""
ENV COLLECTION_NAME=""
ENV QDRANT_SEARCH_LIMIT="10"
ENV QDRANT_READ_ONLY="false"

# --- Chunking ---
ENV CHUNK_SIZE="1500"
ENV CHUNK_OVERLAP="375"

# Run combined MCP + REST API server
CMD ["mcp-server-qdrant", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]