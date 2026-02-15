"""
Summary module for generating LLM-based document abstracts.

Uses Ollama to generate a concise summary of a full document text.
The abstract is stored on every chunk to enable document-level context
without a second lookup.

Disabled when SUMMARY_MODEL is not set (None).
"""

import logging

import httpx

from mcp_server_qdrant.settings import EmbeddingProviderSettings, SummarySettings

logger = logging.getLogger(__name__)

# System prompt for abstract generation
SUMMARY_SYSTEM_PROMPT = (
    "You are a summarization assistant. Generate a concise abstract (2-4 sentences) "
    "of the following document. The abstract should capture the key topic, main points, "
    "and purpose of the document. Write in the same language as the document. "
    "Respond with ONLY the abstract, no preamble or explanation."
)

# System prompt for tag generation
TAGS_SYSTEM_PROMPT = (
    "You are a tagging assistant. Generate 3-6 concise, lowercase tags for the following document. "
    "Tags should capture the key topics, technologies, and concepts. "
    "Use single words or short hyphenated phrases (e.g. 'docker', 'network-config', 'backup'). "
    "Respond with ONLY a comma-separated list of tags, nothing else. "
    "Example: docker, networking, linux, firewall"
)


class SummaryProvider:
    """
    Generates document abstracts via Ollama.

    :param settings: Summary configuration (model name, provider).
    :param ollama_url: Ollama API base URL (reused from embedding provider settings).
    """

    def __init__(
        self,
        settings: SummarySettings,
        ollama_url: str = "http://localhost:11434",
    ):
        self._settings = settings
        self._ollama_url = ollama_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        """Whether summary generation is enabled (model is configured)."""
        return self._settings.summary_model is not None

    async def generate_abstract(self, text: str, title: str | None = None) -> str | None:
        """
        Generate an abstract for the given document text.

        :param text: The full document text.
        :param title: Optional document title for additional context.
        :return: The generated abstract, or None if summarization is disabled or fails.
        """
        if not self.enabled:
            return None

        prompt = text
        if title:
            prompt = f"Title: {title}\n\n{text}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._ollama_url}/api/generate",
                    json={
                        "model": self._settings.summary_model,
                        "system": SUMMARY_SYSTEM_PROMPT,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()
                abstract = data.get("response", "").strip()

                if abstract:
                    logger.debug(
                        "Generated abstract (%d chars) for document '%s'",
                        len(abstract),
                        title or "(untitled)",
                    )
                    return abstract

                logger.warning("Empty abstract returned for document '%s'", title or "(untitled)")
                return None

        except httpx.HTTPStatusError as e:
            logger.error("Ollama API error generating abstract: %s", e)
            return None
        except httpx.ConnectError as e:
            logger.error("Cannot connect to Ollama at %s: %s", self._ollama_url, e)
            return None
        except Exception as e:
            logger.error("Unexpected error generating abstract: %s", e)
            return None

    async def generate_tags(self, text: str, title: str | None = None) -> list[str]:
        """
        Generate tags for the given document text.

        :param text: The full document text.
        :param title: Optional document title for additional context.
        :return: List of generated tags, or empty list if disabled or fails.
        """
        if not self.enabled:
            return []

        prompt = text
        if title:
            prompt = f"Title: {title}\n\n{text}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._ollama_url}/api/generate",
                    json={
                        "model": self._settings.summary_model,
                        "system": TAGS_SYSTEM_PROMPT,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()
                raw = data.get("response", "").strip()

                if raw:
                    tags = [
                        tag.strip().lower().strip('"\'')
                        for tag in raw.split(",")
                        if tag.strip()
                    ]
                    # Filter out overly long or empty tags
                    tags = [t for t in tags if 0 < len(t) <= 40]
                    logger.debug(
                        "Generated %d tags for document '%s': %s",
                        len(tags),
                        title or "(untitled)",
                        tags,
                    )
                    return tags

                logger.warning("Empty tags returned for document '%s'", title or "(untitled)")
                return []

        except httpx.HTTPStatusError as e:
            logger.error("Ollama API error generating tags: %s", e)
            return []
        except httpx.ConnectError as e:
            logger.error("Cannot connect to Ollama at %s: %s", self._ollama_url, e)
            return []
        except Exception as e:
            logger.error("Unexpected error generating tags: %s", e)
            return []


def create_summary_provider(
    summary_settings: SummarySettings,
    embedding_provider_settings: EmbeddingProviderSettings | None = None,
) -> SummaryProvider:
    """
    Factory function to create a SummaryProvider.

    Reuses the Ollama URL from embedding provider settings if available,
    otherwise defaults to localhost.

    :param summary_settings: Summary configuration.
    :param embedding_provider_settings: Optional embedding settings to reuse ollama_url.
    :return: A configured SummaryProvider instance.
    """
    ollama_url = "http://localhost:11434"
    if embedding_provider_settings:
        ollama_url = embedding_provider_settings.ollama_url

    return SummaryProvider(settings=summary_settings, ollama_url=ollama_url)
