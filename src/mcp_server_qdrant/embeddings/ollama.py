import httpx

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama implementation of the embedding provider.

    :param model_name: The name of the Ollama embedding model (e.g., 'embeddinggemma', 'nomic-embed-text').
    :param ollama_url: The base URL of the Ollama API (default: http://localhost:11434).
    """

    def __init__(self, model_name: str, ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self._vector_size: int | None = None

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        embeddings = []
        for doc in documents:
            embedding = await self._get_embedding(doc)
            embeddings.append(embedding)
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        return await self._get_embedding(query)

    def get_vector_name(self) -> str:
        """Return the name of the vector for the Qdrant collection."""
        model_name = self.model_name.replace("/", "-").replace(":", "-").lower()
        return f"ollama-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        if self._vector_size is None:
            # Get vector size by doing a test embedding synchronously
            response = httpx.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": "test"},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            self._vector_size = len(data["embedding"])
        return self._vector_size