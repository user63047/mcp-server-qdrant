# mcp-server-qdrant (Fork with Ollama & Extended Tools)

Fork of the official [Qdrant MCP Server](https://github.com/qdrant/mcp-server-qdrant) with added Ollama embedding support, full CRUD operations, tag management, and intelligent access tracking with automated cleanup.

## What's Different from the Original?

- **Ollama as Embedding Provider** – Use local Ollama models instead of cloud-based embeddings
- **Full CRUD Operations** – Update, delete, and manage entries (not just store & find)
- **Collection Management** – List all available collections
- **Tag Management** – Add and remove tags granularly
- **Access Tracking** – Automatic relevance scoring based on usage patterns
- **Cleanup Tool** – CLI tool to remove stale entries using exponential time decay

## Tools

| Tool | Description |
|------|-------------|
| `qdrant-store` | Store information with optional metadata, tags, and category |
| `qdrant-find` | Semantic search across entries (updates relevance score +3) |
| `qdrant-list` | List entries, optionally filtered (updates relevance score +1) |
| `qdrant-collections` | List all available collections |
| `qdrant-update` | Update content with new embeddings, preserves metadata (updates relevance score +2) |
| `qdrant-set-metadata` | Change metadata without regenerating embeddings |
| `qdrant-add-tags` | Add tags without removing existing ones |
| `qdrant-remove-tags` | Remove specific tags |
| `qdrant-delete` | Delete entries based on filter criteria |

## Filter Functionality

All tools that use filters (`delete`, `update`, `set-metadata`, `add-tags`, `remove-tags`, `list`) support:

| Filter Type | Example | Description |
|-------------|---------|-------------|
| Metadata (exact) | `{"category": "homelab"}` | Exact match on metadata fields |
| Content (substring) | `{"content": "server"}` | Substring search in content |
| Combined | `{"category": "homelab", "content": "server"}` | AND combination of both |
| Tags | `{"tags": ["docker", "backup"]}` | Finds entries with any of the tags |

## Access Tracking & Relevance Score

Entries automatically track how often and how recently they are accessed, similar to how human memory works – frequently used information stays relevant, unused information fades over time.

### How It Works

Each entry stores two tracking fields in its metadata:

- **`relevance_score`** – Weighted access counter (initialized at 0)
- **`last_accessed_at`** – Timestamp of the last access

Different actions contribute different weights to the relevance score:

| Action | Score Increment | Reasoning |
|--------|----------------|-----------|
| `qdrant-find` | +3 | Strongest signal – actively searched and found |
| `qdrant-update` | +2 | Content was actively maintained |
| `qdrant-list` | +1 | Browsing, less intentional than searching |

### Metadata Structure

```json
{
  "document": "The actual text content",
  "metadata": {
    "category": "example",
    "tags": ["tag1", "tag2"],
    "source": "chat",
    "created_at": "2026-01-25T...",
    "updated_at": "2026-01-25T...",
    "last_accessed_at": "2026-02-01T...",
    "relevance_score": 28
  }
}
```

Automatic timestamps:
- `created_at` – Set automatically on `qdrant-store`
- `updated_at` – Updated on `qdrant-update`, `qdrant-set-metadata`, `qdrant-add-tags`, `qdrant-remove-tags`
- `last_accessed_at` – Updated on `qdrant-find`, `qdrant-list`, `qdrant-update`

## Cleanup Tool

The `qdrant-cleanup` CLI tool identifies and removes stale entries using exponential time decay.

### Decay Formula

```
effective_score = relevance_score × e^(-λ × days_since_last_access)
```

With default λ = 0.001, scores decay over time like this:

| Days Since Last Access | Decay Factor | Score 50 → Effective | Score 5 → Effective |
|------------------------|-------------|---------------------|-------------------|
| 0 (today) | 1.0 | 50.0 | 5.0 |
| 365 (1 year) | 0.69 | 34.7 | 3.5 |
| 1095 (3 years) | 0.33 | 16.7 | 1.7 |
| 1825 (5 years) | 0.16 | 8.1 | 0.8 |

### Usage

```bash
# Dry run – only show what would be deleted
qdrant-cleanup --dry-run --qdrant-url http://localhost:6333 --qdrant-api-key your_key

# Delete entries below threshold
qdrant-cleanup --qdrant-url http://localhost:6333 --qdrant-api-key your_key

# Custom threshold and decay rate
qdrant-cleanup --threshold 3.0 --decay-lambda 0.002 --qdrant-url http://localhost:6333

# Only process a specific collection
qdrant-cleanup --dry-run --collection my-collection --qdrant-url http://localhost:6333
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | off | Only report, don't delete |
| `--threshold` | 1.0 | Effective score below which entries are deleted |
| `--decay-lambda` | 0.001 | Decay rate (higher = faster decay) |
| `--collection` | all | Only process a specific collection |
| `--qdrant-url` | `$QDRANT_URL` | Qdrant server URL |
| `--qdrant-api-key` | `$QDRANT_API_KEY` | Qdrant API key |

### Automating Cleanup with Cron

```bash
# Run cleanup monthly (dry-run example)
0 0 1 * * /path/to/venv/bin/qdrant-cleanup --qdrant-url http://localhost:6333 --qdrant-api-key your_key
```

## Prerequisites

- **Ollama** with an embedding model (e.g. `embeddinggemma:300m`)
- **Qdrant** vector database
- **Python** 3.10+
- **uv** (Python package manager)

## Installation

### Install from GitHub

```bash
uv venv ~/mcp-qdrant-venv
source ~/mcp-qdrant-venv/bin/activate
uv pip install git+https://github.com/user63047/mcp-server-qdrant.git
```

This installs both commands:
- `mcp-server-qdrant` – The MCP server
- `qdrant-cleanup` – The cleanup tool

### Update to Latest Version

```bash
source ~/mcp-qdrant-venv/bin/activate
uv pip install --upgrade git+https://github.com/user63047/mcp-server-qdrant.git
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_PROVIDER` | `fastembed` or `ollama` | `fastembed` |
| `EMBEDDING_MODEL` | Model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `OLLAMA_URL` | Ollama API URL | `http://localhost:11434` |
| `QDRANT_URL` | Qdrant API URL | None |
| `QDRANT_API_KEY` | Qdrant API key (if enabled) | None |
| `COLLECTION_NAME` | Default Qdrant collection | None |

### Start Server Manually

```bash
EMBEDDING_PROVIDER=ollama \
EMBEDDING_MODEL=embeddinggemma:300m \
OLLAMA_URL=http://localhost:11434 \
QDRANT_URL=http://localhost:6333 \
QDRANT_API_KEY=your_api_key \
COLLECTION_NAME=my-collection \
mcp-server-qdrant
```

### MCP Client Integration (STDIO/JSON)

For MCP clients like Msty Studio, Claude Desktop, or similar:

```json
{
  "command": "/path/to/mcp-qdrant-venv/bin/mcp-server-qdrant",
  "args": [],
  "env": {
    "EMBEDDING_PROVIDER": "ollama",
    "EMBEDDING_MODEL": "embeddinggemma:300m",
    "OLLAMA_URL": "http://localhost:11434",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_API_KEY": "your_api_key",
    "COLLECTION_NAME": "my-collection"
  }
}
```

## Usage Examples

```
# Store with metadata
"Store in Qdrant: 'My server runs Ubuntu 24.04' with metadata category='infrastructure' and tags=['server', 'linux']"

# Semantic search
"Search in Qdrant for 'server operating system'"

# List entries
"List all entries in the Qdrant collection"

# Update content
"Update the entry with category='infrastructure' to new text: 'My server runs Ubuntu 24.04 LTS with 64GB RAM'"

# Manage tags
"Add the tag 'production' to entries with category='infrastructure'"
"Remove the tag 'test' from entries with category='infrastructure'"

# Change metadata
"Set the category to 'archived' for entries with category='old-project'"

# Filter by content
"Delete all entries containing 'test' in the content"

# Combined filter
"Add the tag 'important' to entries with category='infrastructure' that contain 'production' in the content"
```

## Changed Files (vs. Original)

| File | Change |
|------|--------|
| `src/mcp_server_qdrant/embeddings/ollama.py` | New – Ollama provider implementation |
| `src/mcp_server_qdrant/embeddings/types.py` | Added OLLAMA enum |
| `src/mcp_server_qdrant/embeddings/factory.py` | Ollama provider factory |
| `src/mcp_server_qdrant/settings.py` | `ollama_url` setting + new tool descriptions |
| `src/mcp_server_qdrant/qdrant.py` | CRUD methods + access tracking |
| `src/mcp_server_qdrant/mcp_server.py` | New tools registered |
| `src/mcp_server_qdrant/cleanup.py` | New – Cleanup CLI tool |
| `pyproject.toml` | httpx dependency + cleanup entrypoint |

## Troubleshooting

**Error: 401 Unauthorized**
Qdrant has authentication enabled. Add `QDRANT_API_KEY` to your configuration.

**Server starts without output**
MCP servers communicate via stdio – this is normal. They wait for input from an MCP client.

**Filter finds no entries**
- Content filter uses substring search (`{"content": "server"}` finds "My server runs...")
- Metadata filter uses exact match (`{"category": "homelab"}` only matches exactly "homelab")
- Tags filter: `{"tags": ["docker"]}` finds entries that have "docker" in their tags array

**Entries without tracking data**
Entries created before the access tracking feature was added won't have `relevance_score` or `last_accessed_at`. The cleanup tool skips these entries by default.

## Acknowledgments

This fork was developed with the assistance of [Claude Opus 4.6](https://www.anthropic.com/claude) by Anthropic.

## License

Apache License 2.0 – see [LICENSE](LICENSE) for details.

Original repo: [qdrant/mcp-server-qdrant](https://github.com/qdrant/mcp-server-qdrant)
