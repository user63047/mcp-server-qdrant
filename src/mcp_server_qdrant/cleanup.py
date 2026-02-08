import argparse
import math
import os
import sys
from datetime import datetime, timezone

from qdrant_client import QdrantClient


def calculate_effective_score(relevance_score: float, days_since_access: float, decay_lambda: float) -> float:
    """
    Calculate the effective score with exponential decay.
    effective_score = relevance_score × e^(-λ × days)
    """
    return relevance_score * math.exp(-decay_lambda * days_since_access)


def main():
    """
    Cleanup tool for Qdrant collections.
    Calculates effective scores using exponential decay and removes
    or reports entries below the threshold.
    """
    parser = argparse.ArgumentParser(
        description="Qdrant Cleanup - Remove entries with low relevance based on access tracking and time decay"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report entries that would be deleted, without actually deleting them",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Effective score below which entries are deleted (default: 1.0)",
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
    total_no_tracking = 0

    print(f"{'=' * 60}")
    print(f"Qdrant Cleanup {'(DRY RUN)' if args.dry_run else ''}")
    print(f"Threshold: {args.threshold} | Lambda: {args.decay_lambda}")
    print(f"Date: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 60}")

    for collection_name in collection_names:
        print(f"\n--- Collection: {collection_name} ---")

        # Scroll through all entries
        offset = None
        entries_to_delete = []
        entries_kept = 0
        entries_no_tracking = 0

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
                metadata = point.payload.get("metadata", {})
                relevance_score = metadata.get("relevance_score")
                last_accessed_at = metadata.get("last_accessed_at")

                # Skip entries without access tracking
                if relevance_score is None or last_accessed_at is None:
                    entries_no_tracking += 1
                    continue

                # Calculate days since last access
                try:
                    last_access = datetime.fromisoformat(last_accessed_at)
                    days_since_access = (now - last_access).total_seconds() / 86400
                except (ValueError, TypeError):
                    entries_no_tracking += 1
                    continue

                # Calculate effective score
                effective_score = calculate_effective_score(
                    relevance_score, days_since_access, args.decay_lambda
                )

                if effective_score < args.threshold:
                    content_preview = point.payload.get("document", "")[:80]
                    entries_to_delete.append({
                        "id": point.id,
                        "content_preview": content_preview,
                        "relevance_score": relevance_score,
                        "days_since_access": round(days_since_access, 1),
                        "effective_score": round(effective_score, 2),
                    })
                else:
                    entries_kept += 1

            if next_offset is None:
                break
            offset = next_offset

        # Report
        if entries_to_delete:
            print(f"\n  Entries below threshold ({args.threshold}):")
            for entry in entries_to_delete:
                print(
                    f"    Score: {entry['relevance_score']} → {entry['effective_score']} "
                    f"(after {entry['days_since_access']} days) | "
                    f"\"{entry['content_preview']}...\""
                )

            if not args.dry_run:
                ids_to_delete = [entry["id"] for entry in entries_to_delete]
                client.delete(
                    collection_name=collection_name,
                    points_selector=ids_to_delete,
                )
                print(f"\n  ✓ Deleted {len(entries_to_delete)} entries")
            else:
                print(f"\n  → Would delete {len(entries_to_delete)} entries")
        else:
            print("  No entries below threshold.")

        print(f"  Kept: {entries_kept} | No tracking data: {entries_no_tracking}")

        total_deleted += len(entries_to_delete)
        total_kept += entries_kept
        total_no_tracking += entries_no_tracking

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  {'Would delete' if args.dry_run else 'Deleted'}: {total_deleted}")
    print(f"  Kept: {total_kept}")
    print(f"  No tracking data: {total_no_tracking}")
    print(f"{'=' * 60}")
