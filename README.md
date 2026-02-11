# mcp-server-qdrant (Fork with Document Chunking, Ollama & Extended Tools)

> ⚠️ **Work in Progress** – This fork is under active development. The two-level document/chunk architecture described below is functional but not yet considered stable. External source synchronization is planned but not yet implemented. APIs and payload structures may change.

Fork of the official [Qdrant MCP Server](https://github.com/qdrant/mcp-server-qdrant) with a **two-level document/chunk architecture**, Ollama embedding support, full CRUD operations, LLM-generated abstracts, tag management, and intelligent access tracking with automated cleanup.

## What's Different from the Original?

- **Two-Level Architecture** – Documents are automatically split into embedding-friendly chunks, solving the silent truncation problem when texts exceed the embedding model's context window
- **Document-Level Operations** – All tools operate on documents (grouped by `document_id`), not individual vector points
- **Source Type Distinction** – Differentiates between `composed` entries (created and managed in Qdrant) and external sources (indexed from other systems via sync)
- **LLM-Generated Abstracts** – Optional document summaries via Ollama, stored on every chunk for fast retrieval
- **Ollama as Embedding Provider** – Use local Ollama models instead of cloud-based embeddings
- **Full CRUD Operations** – Store, find, update, append, delete, and manage documents
- **Disambiguation** – When multiple documents match a filter, they are returned for the LLM to ask the user which one is meant
- **Access Tracking & Cleanup** – Automatic relevance scoring with exponential time decay

## Architecture Overview

Embedding models have a limited context window (e.g. ~2048 tokens). Texts exceeding this limit are silently truncated during embedding — information is lost without warning. This fork solves this by splitting documents into overlapping chunks that each fit the embedding model's context window. All chunks share a `document_id` and carry redundant document-level metadata.

### Source Types

| Source Type | Description | Content Operations | Metadata Operations |
|---|---|---|---|
| `composed` | Content created/managed in Qdrant | Read/Write | Read/Write |
| External | Content indexed from other systems | Read-only | Read/Write |

External sources are indexed into Qdrant for search. Content changes must be made at the source; a sync layer updates Qdrant automatically. Metadata (tags, category) can be changed directly in Qdrant for all source types. Custom `source_type` values can be used to distinguish different external systems.

> **Note:** Synchronization with external sources is not yet implemented. Currently, only `composed` entries are fully functional.

## Tools

| Tool | Description | Source Types |
|---|---|---|
| `qdrant-store` | Store a document with title, content, and metadata | All |
| `qdrant-find` | Semantic search, returns document-level results | All |
| `qdrant-list` | List documents, optionally filtered by metadata | All |
| `qdrant-collections` | List all available collections | — |
| `qdrant-update` | Replace document content (re-chunks, re-embeds) | Composed only |
| `qdrant-append` | Append text to existing document | Composed only |
| `qdrant-set-metadata` | Update metadata without changing content | All |
| `qdrant-add-tags` | Add tags without removing existing ones | All |
| `qdrant-remove-tags` | Remove specific tags | All |
| `qdrant-delete` | Delete a document and all its chunks | Composed only |

### Disambiguation

When `update`, `append`, or `delete` match multiple documents, no action is performed. Instead, the matching documents are returned (title, abstract, metadata, `document_id`) so the LLM can ask the user which document is meant. A second call with `document_id` targets the exact document.

## Filter Functionality

All tools that accept filters support:

| Filter Type | Example | Description |
|---|---|---|
| Document ID | `{"document_id": "abc123"}` | Exact document targeting |
| Title | `{"title": "Docker"}` | Text match on title |
| Content | `{"content": "server"}` | Substring search in chunk text |
| Category | `{"category": "homelab"}` | Exact match on metadata |
| Tags | `{"tags": ["docker", "backup"]}` | Matches entries with any of the tags |
| Source Type | `{"source_type": "composed"}` | Filter by source type |
| Combined | `{"category": "homelab", "content": "server"}` | AND combination |

## Access Tracking & Relevance Score

Documents track access frequency and recency. Different actions contribute different weights:

| Action | Score Increment |
|---|---|
| `qdrant-find` | +3 |
| `qdrant-update` / `qdrant-append` | +2 |
| `qdrant-list` | +1 |

Access tracking updates all chunks of a document to keep metadata synchronized.

## Cleanup Tool

The `qdrant-cleanup` CLI tool removes stale `composed` documents using exponential time decay. External sources are skipped — they are managed by their respective sync pipelines.

```
effective_score = relevance_score × e^(-λ × days_since_last_access)
```

Cleanup groups points by `document_id` and evaluates at the document level — a document is either fully deleted (all chunks) or kept.

```bash
# Dry run
qdrant-cleanup --dry-run --qdrant-url http://localhost:6333

# Delete documents below threshold
qdrant-cleanup --qdrant-url http://localhost:6333 --qdrant-api-key your_key

# Custom threshold and decay rate
qdrant-cleanup --threshold 3.0 --decay-lambda 0.002 --qdrant-url http://localhost:6333
```

| Flag | Default | Description |
|---|---|---|
| `--dry-run` | off | Only report, don't delete |
| `--threshold` | 1.0 | Effective score below which documents are deleted |
| `--decay-lambda` | 0.001 | Decay rate (higher = faster decay) |
| `--collection` | all | Only process a specific collection |
| `--qdrant-url` | `$QDRANT_URL` | Qdrant server URL |
| `--qdrant-api-key` | `$QDRANT_API_KEY` | Qdrant API key |

## Prerequisites

- **Ollama** with an embedding model and optionally a summary model
- **Qdrant** vector database
- **Python** 3.10+

## Installation

```bash
pip install git+https://github.com/user63047/mcp-server-qdrant.git
```

This installs two commands:
- `mcp-server-qdrant` – The MCP server
- `qdrant-cleanup` – The cleanup CLI tool

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed` or `ollama` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model name |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `QDRANT_URL` | — | Qdrant server URL |
| `QDRANT_API_KEY` | — | Qdrant API key |
| `COLLECTION_NAME` | — | Default Qdrant collection |
| `QDRANT_SEARCH_LIMIT` | `10` | Maximum search results |
| `QDRANT_READ_ONLY` | `false` | Disable write operations |
| `CHUNK_SIZE` | `1500` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | `375` | Overlap between chunks in tokens |
| `SUMMARY_MODEL` | — | Ollama model for abstracts (disabled if not set) |
| `SUMMARY_PROVIDER` | `ollama` | Summary provider |

### MCP Client Configuration

```json
{
  "command": "/path/to/venv/bin/mcp-server-qdrant",
  "args": [],
  "env": {
    "EMBEDDING_PROVIDER": "ollama",
    "EMBEDDING_MODEL": "embeddinggemma:300m",
    "SUMMARY_MODEL": "gemma3:4b",
    "OLLAMA_URL": "http://localhost:11434",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_API_KEY": "your_api_key",
    "COLLECTION_NAME": "my-collection"
  }
}
```

## Troubleshooting

**Entries without document_id** – Entries from a previous version lack the new fields and won't work with the current tools. Delete the old collection and re-index.

**Empty abstracts** – Ensure `SUMMARY_MODEL` is set and the model is available in Ollama. Abstract generation is disabled if not configured.

**Server starts without output** – MCP servers communicate via stdio. They wait for input from an MCP client.

## Acknowledgments

This fork was developed with the assistance of [Claude](https://www.anthropic.com/claude) by Anthropic.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Original repo: [qdrant/mcp-server-qdrant](https://github.com/qdrant/mcp-server-qdrant)
