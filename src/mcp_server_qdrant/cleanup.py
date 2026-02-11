import argparse
import math
import os
import sys
from datetime import datetime, timezone

from qdrant_client import QdrantClient, models


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
    Only processes 'composed' entries — external sources are managed by
    their respective sync pipelines.
    """
    parser = argparse.ArgumentParser(
        description="Qdrant Cleanup - Remove composed documents with low relevance based on access tracking and time decay"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report documents that would be deleted, without actually deleting them",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Effective score below which documents are deleted (default: 1.0)",
    )
    parser.add_argument(
        "--decay-lambda",
        type=float,
        default=0.001,
        help="Decay rate lambda (default: 0.001)",
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
    total_skipped_external = 0

    print(f"{'=' * 60}")
    print(f"Qdrant Cleanup {'(DRY RUN)' if args.dry_run else ''}")
    print(f"Threshold: {args.threshold} | Lambda: {args.decay_lambda}")
    print(f"Mode: Document-level | Only source_type: composed")
    print(f"Date: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 60}")

    for collection_name in collection_names:
        print(f"\n--- Collection: {collection_name} ---")

        # Scroll through all points, grouped by document_id.
        # We only need chunk_index 0 per document for metadata,
        # but we scroll all to count chunks.
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
        docs_to_delete = []
        docs_kept = 0
        docs_external = 0

        for doc_id, doc in documents.items():
            # Only process composed entries
            if doc["source_type"] != "composed":
                docs_external += 1
                continue

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
                relevance_score, days_since_access, args.decay_lambda
            )

            if effective_score < args.threshold:
                docs_to_delete.append({
                    "document_id": doc_id,
                    "title": doc["title"],
                    "chunk_count": doc["chunk_count"],
                    "relevance_score": relevance_score,
                    "days_since_access": round(days_since_access, 1),
                    "effective_score": round(effective_score, 2),
                })
            else:
                docs_kept += 1

        # Report
        if docs_to_delete:
            print(f"\n  Documents below threshold ({args.threshold}):")
            for doc in docs_to_delete:
                print(
                    f"    [{doc['chunk_count']} chunk(s)] "
                    f"Score: {doc['relevance_score']} → {doc['effective_score']} "
                    f"(after {doc['days_since_access']} days) | "
                    f"\"{doc['title']}\""
                )

            if not args.dry_run:
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
            print("  No composed documents below threshold.")

        print(f"  Kept (composed): {docs_kept} | External (skipped): {docs_external}")

        total_deleted += len(docs_to_delete)
        total_kept += docs_kept
        total_skipped_external += docs_external

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  {'Would delete' if args.dry_run else 'Deleted'}: {total_deleted} document(s)")
    print(f"  Kept (composed): {total_kept}")
    print(f"  External (skipped): {total_skipped_external}")
    print(f"{'=' * 60}")
