"""
Summary module for generating LLM-based document abstracts and tags.

Uses Ollama to generate a concise summary of a full document text.
The abstract is stored on every chunk to enable document-level context
without a second lookup.

Tag generation uses the abstract (not full text) for efficiency.
On updates, existing tags are evaluated for relevance before merging.

Disabled when SUMMARY_MODEL is not set (None).
"""

import logging

import httpx

from mcp_server_qdrant.settings import EmbeddingProviderSettings, SummarySettings

logger = logging.getLogger(__name__)

# Maximum number of tags per document
MAX_TAGS = 15

# System prompt for abstract generation
SUMMARY_SYSTEM_PROMPT = (
    "You are a summarization assistant. Generate a concise abstract (2-4 sentences) "
    "of the following document. The abstract should capture the key topic, main points, "
    "and purpose of the document. Write in the same language as the document. "
    "Respond with ONLY the abstract, no preamble or explanation."
)

# System prompt for tag generation (new documents without existing tags)
TAGS_SYSTEM_PROMPT = (
    "You are a tagging assistant. Generate 5-15 concise, lowercase tags for the following document. "
    "Tags should capture the key topics, technologies, and concepts. "
    "Use single words or short hyphenated phrases (e.g. 'docker', 'network-config', 'backup'). "
    "Do NOT use IP addresses, port numbers, file paths, or overly specific technical values as tags. "
    "Respond with ONLY a comma-separated list of tags, nothing else. "
    "Example: docker, networking, linux, firewall, wake-on-lan"
)

# System prompt for tag evaluation (existing documents with tags)
EVALUATE_TAGS_SYSTEM_PROMPT = (
    "You are a tagging assistant. You receive a document abstract, its title, "
    "and a list of existing tags. Your job:\n"
    "1. EVALUATE each existing tag: Is it still relevant to the document content? "
    "Keep relevant ones, drop irrelevant or overly specific ones "
    "(like IP addresses, port numbers, file paths, typos, or duplicates).\n"
    "2. ADD missing tags if important topics from the abstract are not covered.\n"
    "3. Return at most 15 tags total, prioritized by relevance.\n"
    "Use single words or short hyphenated phrases, all lowercase.\n"
    "Respond with ONLY a comma-separated list of tags, nothing else."
)


class SummaryProvider:
    """
    Generates document abstracts and tags via Ollama.

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

    async def generate_tags(
        self, abstract: str, title: str | None = None
    ) -> list[str]:
        """
        Generate tags for a NEW document (no existing tags).
        Uses the abstract for efficiency instead of full text.

        :param abstract: The document abstract (generated by generate_abstract).
        :param title: Optional document title for additional context.
        :return: List of generated tags (max MAX_TAGS), or empty list if disabled/fails.
        """
        if not self.enabled:
            return []

        prompt = abstract
        if title:
            prompt = f"Title: {title}\n\nAbstract: {abstract}"

        return await self._call_ollama_for_tags(
            system_prompt=TAGS_SYSTEM_PROMPT,
            prompt=prompt,
            title=title,
            log_action="Generated",
        )

    async def evaluate_tags(
        self,
        abstract: str,
        title: str | None = None,
        existing_tags: list[str] | None = None,
    ) -> list[str]:
        """
        Evaluate and refresh tags for an EXISTING document.
        Checks existing tags for relevance against the abstract,
        removes irrelevant ones, and adds missing ones.

        If no existing tags are provided, falls back to generate_tags().

        :param abstract: The document abstract.
        :param title: Optional document title.
        :param existing_tags: Current tags on the document.
        :return: Evaluated list of tags (max MAX_TAGS).
        """
        if not self.enabled:
            return existing_tags or []

        # No existing tags → just generate fresh
        if not existing_tags:
            return await self.generate_tags(abstract, title)

        prompt_parts = []
        if title:
            prompt_parts.append(f"Title: {title}")
        prompt_parts.append(f"Abstract: {abstract}")
        prompt_parts.append(f"Existing tags: {', '.join(existing_tags)}")
        prompt = "\n\n".join(prompt_parts)

        result = await self._call_ollama_for_tags(
            system_prompt=EVALUATE_TAGS_SYSTEM_PROMPT,
            prompt=prompt,
            title=title,
            log_action="Evaluated",
        )

        # Fallback: if evaluation returns nothing, keep existing (capped)
        if not result:
            logger.warning(
                "Tag evaluation returned empty for '%s', keeping existing tags (capped to %d)",
                title or "(untitled)",
                MAX_TAGS,
            )
            return existing_tags[:MAX_TAGS]

        return result

    async def _call_ollama_for_tags(
        self,
        system_prompt: str,
        prompt: str,
        title: str | None = None,
        log_action: str = "Processed",
    ) -> list[str]:
        """
        Shared helper for calling Ollama to generate/evaluate tags.

        :param system_prompt: The system prompt to use.
        :param prompt: The user prompt to send.
        :param title: Document title for logging.
        :param log_action: Verb for log messages (e.g. 'Generated', 'Evaluated').
        :return: Parsed and cleaned list of tags.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._ollama_url}/api/generate",
                    json={
                        "model": self._settings.summary_model,
                        "system": system_prompt,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()
                raw = data.get("response", "").strip()

                if raw:
                    tags = self._parse_tags(raw)
                    logger.debug(
                        "%s %d tags for document '%s': %s",
                        log_action,
                        len(tags),
                        title or "(untitled)",
                        tags,
                    )
                    return tags

                logger.warning(
                    "Empty tag response for document '%s'", title or "(untitled)"
                )
                return []

        except httpx.HTTPStatusError as e:
            logger.error("Ollama API error for tags: %s", e)
            return []
        except httpx.ConnectError as e:
            logger.error("Cannot connect to Ollama at %s: %s", self._ollama_url, e)
            return []
        except Exception as e:
            logger.error("Unexpected error for tags: %s", e)
            return []

    @staticmethod
    def _parse_tags(raw: str) -> list[str]:
        """
        Parse and clean raw LLM tag output.

        Handles comma-separated output, strips quotes and whitespace,
        removes duplicates, and enforces MAX_TAGS limit.

        :param raw: Raw LLM response string.
        :return: Cleaned, deduplicated, capped list of tags.
        """
        tags = [
            tag.strip().lower().strip("\"'`")
            for tag in raw.split(",")
            if tag.strip()
        ]
        # Filter out overly long, empty, or junk tags
        tags = [t for t in tags if 0 < len(t) <= 40]
        # Deduplicate while preserving order (LLM puts most relevant first)
        seen: set[str] = set()
        unique_tags: list[str] = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        # Enforce maximum
        return unique_tags[:MAX_TAGS]


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
