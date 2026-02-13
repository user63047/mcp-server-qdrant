FROM python:3.11-slim

WORKDIR /app

# Install git (required for pip install from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install --no-cache-dir uv

# Install mcp-server-qdrant from GitHub fork
RUN uv pip install --system --no-cache-dir \
    "mcp-server-qdrant @ git+https://github.com/user63047/mcp-server-qdrant.git@feature/container-deployment"

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
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME=""
ENV QDRANT_SEARCH_LIMIT="10"
ENV QDRANT_READ_ONLY="false"

# --- Chunking ---
ENV CHUNK_SIZE="1500"
ENV CHUNK_OVERLAP="375"

# Run combined MCP + REST API server
CMD ["mcp-server-qdrant", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]