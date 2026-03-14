import argparse
import math
import os
import sys
from datetime import datetime, timezone

from qdrant_client import QdrantClient, models

from mcp_server_qdrant.settings import CleanupSettings


def calculate_effective_score(relevance_score: float, days_since_access: float, decay_lambda: float) -> float:
    """
    Calculate the effective score with exponential decay.
    effective_score = relevance_score × e^(-λ × days)
    """
    return relevance_score * math.exp(-decay_lambda * days_since_access)


def main():
    """
    Cleanup tool for Qdrant collections (two-level document/chunk model).

    Groups chunks by document_id and evaluates at the document level.
    Processes source types configured via CLEANUP_SOURCE_TYPES env var
    or --source-types CLI argument. Each source type has its own threshold.
    Source types not listed are skipped entirely.
    """

    # Load settings from environment variables (defaults apply if not set)
    settings = CleanupSettings()

    parser = argparse.ArgumentParser(
        description=(
            "Qdrant Cleanup - Remove documents with low relevance based on "
            "access tracking and time decay. Source types and thresholds are "
            "configured via CLEANUP_SOURCE_TYPES env var or --source-types."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Only report documents that would be deleted, without actually deleting them",
    )
    parser.add_argument(
        "--source-types",
        type=str,
        default=None,
        help=(
            "Comma-separated source_type:threshold pairs. "
            "Overrides CLEANUP_SOURCE_TYPES env var. "
            "Example: 'composed:1.0,trilium:1.0,paperless:0.3'"
        ),
    )
    parser.add_argument(
        "--decay-lambda",
        type=float,
        default=None,
        help="Decay rate lambda (overrides CLEANUP_DECAY_LAMBDA env var)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Only process this collection (default: all collections)",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant URL (default: from QDRANT_URL env var)",
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=None,
        help="Qdrant API key (default: from QDRANT_API_KEY env var)",
    )
    args = parser.parse_args()

    # Resolve configuration: CLI args > env vars > defaults
    dry_run = args.dry_run if args.dry_run is not None else settings.dry_run
    decay_lambda = args.decay_lambda if args.decay_lambda is not None else settings.decay_lambda

    # Parse source types with per-type thresholds
    if args.source_types is not None:
        # CLI override: parse manually (same format as env var)
        source_type_thresholds: dict[str, float] = {}
        for entry in args.source_types.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                st, th = entry.split(":", 1)
                source_type_thresholds[st.strip()] = float(th.strip())
            else:
                source_type_thresholds[entry.strip()] = 1.0
    else:
        source_type_thresholds = settings.parse_source_types()

    if not source_type_thresholds:
        print("Error: No source types configured. Set CLEANUP_SOURCE_TYPES or use --source-types.")
        sys.exit(1)

    # Resolve Qdrant connection
    qdrant_url = args.qdrant_url or os.environ.get("QDRANT_URL")
    qdrant_api_key = args.qdrant_api_key or os.environ.get("QDRANT_API_KEY")

    if not qdrant_url:
        print("Error: No Qdrant URL provided. Use --qdrant-url or set QDRANT_URL env var.")
        sys.exit(1)

    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Determine which collections to process
    if args.collection:
        collection_names = [args.collection]
    else:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

    if not collection_names:
        print("No collections found.")
        return

    now = datetime.now(timezone.utc)
    total_deleted = 0
    total_kept = 0
    total_skipped = 0

    # Format source types for display
    st_display = ", ".join(f"{st}(≤{th})" for st, th in source_type_thresholds.items())

    print(f"{'=' * 60}")
    print(f"Qdrant Cleanup {'(DRY RUN)' if dry_run else ''}")
    print(f"Source types: {st_display}")
    print(f"Lambda: {decay_lambda}")
    print(f"Mode: Document-level grouping")
    print(f"Date: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 60}")

    for collection_name in collection_names:
        print(f"\n--- Collection: {collection_name} ---")

        # Scroll through all points, grouped by document_id.
        offset = None
        # doc_id -> {title, relevance_score, last_accessed_at, source_type, chunk_count}
        documents: dict[str, dict] = {}

        while True:
            results, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not results:
                break

            for point in results:
                payload = point.payload
                doc_id = payload.get("document_id")

                if not doc_id:
                    # Legacy point without document_id — skip
                    continue

                if doc_id not in documents:
                    metadata = payload.get("metadata", {})
                    documents[doc_id] = {
                        "title": payload.get("title", "(untitled)"),
                        "source_type": metadata.get("source_type", "unknown"),
                        "relevance_score": metadata.get("relevance_score"),
                        "last_accessed_at": metadata.get("last_accessed_at"),
                        "chunk_count": 1,
                    }
                else:
                    documents[doc_id]["chunk_count"] += 1

            if next_offset is None:
                break
            offset = next_offset

        # Evaluate each document
        docs_to_delete: list[dict] = []
        docs_kept = 0
        docs_skipped = 0

        for doc_id, doc in documents.items():
            source_type = doc["source_type"]

            # Only process configured source types — skip everything else
            if source_type not in source_type_thresholds:
                docs_skipped += 1
                continue

            threshold = source_type_thresholds[source_type]
            relevance_score = doc["relevance_score"]
            last_accessed_at = doc["last_accessed_at"]

            # Skip documents without access tracking
            if relevance_score is None or last_accessed_at is None:
                docs_kept += 1
                continue

            # Calculate days since last access
            try:
                last_access = datetime.fromisoformat(last_accessed_at)
                days_since_access = (now - last_access).total_seconds() / 86400
            except (ValueError, TypeError):
                docs_kept += 1
                continue

            # Calculate effective score
            effective_score = calculate_effective_score(
                relevance_score, days_since_access, decay_lambda
            )

            if effective_score < threshold:
                docs_to_delete.append({
                    "document_id": doc_id,
                    "title": doc["title"],
                    "source_type": source_type,
                    "chunk_count": doc["chunk_count"],
                    "relevance_score": relevance_score,
                    "threshold": threshold,
                    "days_since_access": round(days_since_access, 1),
                    "effective_score": round(effective_score, 4),
                })
            else:
                docs_kept += 1

        # Report
        if docs_to_delete:
            print(f"\n  Documents below their threshold:")
            for doc in docs_to_delete:
                print(
                    f"    [{doc['source_type']}] [{doc['chunk_count']} chunk(s)] "
                    f"Score: {doc['relevance_score']} → {doc['effective_score']} "
                    f"(threshold: {doc['threshold']}, after {doc['days_since_access']} days) | "
                    f"\"{doc['title']}\""
                )

            if not dry_run:
                for doc in docs_to_delete:
                    # Delete all chunks of this document
                    doc_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=doc["document_id"]),
                            )
                        ]
                    )
                    client.delete(
                        collection_name=collection_name,
                        points_selector=models.FilterSelector(filter=doc_filter),
                    )
                print(f"\n  ✓ Deleted {len(docs_to_delete)} document(s)")
            else:
                print(f"\n  → Would delete {len(docs_to_delete)} document(s)")
        else:
            print("  No documents below threshold.")

        print(f"  Kept: {docs_kept} | Skipped (unconfigured types): {docs_skipped}")

        total_deleted += len(docs_to_delete)
        total_kept += docs_kept
        total_skipped += docs_skipped

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  {'Would delete' if dry_run else 'Deleted'}: {total_deleted} document(s)")
    print(f"  Kept: {total_kept}")
    print(f"  Skipped (unconfigured types): {total_skipped}")
    print(f"{'=' * 60}")
