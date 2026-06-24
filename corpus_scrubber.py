# filename: corpus_scrubber.py
"""
Corpus-level deduplication and noise scrubber for meta_reflections (and optionally
other collections). Targets three tiers of garbage:

  Tier 1 — Hard delete: reflection_confidence == 0 AND token_count <= 20
            (zero signal, micro-chunk — pure noise)

  Tier 2 — Soft delete: token_count <= 20 AND confidence > 0
            (micro-chunk that produced some claims but is still junk-grade)

  Tier 3 — Hash dedup: duplicate source_hash values — keep the point with
            the highest reflection_confidence; delete the rest.

Run with --dry-run first to see exactly what would be deleted before touching anything.

Usage:
    python corpus_scrubber.py --dry-run
    python corpus_scrubber.py --dry-run --source thetanakh
    python corpus_scrubber.py --source thetanakh --tier 1
    python corpus_scrubber.py --source thetanakh --tier 1 --tier 2 --tier 3
    python corpus_scrubber.py --source thetanakh --all-tiers
    python corpus_scrubber.py --all-sources --all-tiers
"""

import os
import argparse
import sys
from collections import defaultdict
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

load_dotenv()

# ----------------------------------------------------------------- config ----

QDRANT_URL         = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = "meta_reflections"
SCROLL_BATCH       = 500

# Tier thresholds — tweak if needed
TIER1_MAX_TOKENS = 20   # token_count <= this AND confidence == 0  → hard delete
TIER2_MAX_TOKENS = 20   # token_count <= this AND confidence > 0   → soft delete

# ----------------------------------------------------------------- client ----

client = QdrantClient(url=QDRANT_URL)


# --------------------------------------------------------------- helpers ----

def scroll_all(collection: str, source_filter: str | None = None) -> list:
    """Scroll every point in the collection (optionally filtered by source_file)."""
    must = []
    if source_filter:
        must.append(FieldCondition(key="source_file", match=MatchValue(value=source_filter)))

    filt = Filter(must=must) if must else None
    points = []
    offset = None

    while True:
        result, offset = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(result)
        if offset is None:
            break

    return points


def get_confidence(p) -> float:
    return float(p.payload.get("reflection_confidence", 0) or 0)


def get_token_count(p) -> int:
    return int(p.payload.get("token_count", 0) or 0)


def get_source_hash(p) -> str:
    return p.payload.get("source_hash", "")


def delete_points(collection: str, ids: list, dry_run: bool, label: str) -> int:
    if not ids:
        print(f"  {label}: nothing to delete")
        return 0
    print(f"  {'[DRY RUN] Would delete' if dry_run else 'Deleting'} {len(ids):,} points — {label}")
    if not dry_run:
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            client.delete(
                collection_name=collection,
                points_selector=batch,
            )
        print(f"  ✓ Done")
    return len(ids)


# --------------------------------------------------------------- tiers -----

def is_raw_chunk(p) -> bool:
    """True if this reflection came from a raw mb_chunks source (not mb_claims).
    Only raw chunks should have the token threshold applied — mb_claims reflections
    have small token_counts by design (the claim string itself is short).
    """
    return p.payload.get("source_collection", "") == "mb_chunks"


def tier1_hard_delete(points: list, dry_run: bool, collection: str) -> set:
    """Zero-confidence + micro raw-chunk: no signal at all. Returns deleted IDs."""
    targets = [
        p.id for p in points
        if is_raw_chunk(p)
        and get_confidence(p) == 0.0
        and get_token_count(p) <= TIER1_MAX_TOKENS
    ]
    delete_points(collection, targets, dry_run, "Tier 1 — zero-confidence micro raw-chunks")
    return set(targets)


def tier2_soft_delete(points: list, exclude_ids: set, dry_run: bool, collection: str) -> set:
    """Micro raw-chunks with some confidence — still too small to be meaningful."""
    targets = [
        p.id for p in points
        if p.id not in exclude_ids
        and is_raw_chunk(p)
        and get_confidence(p) > 0.0
        and get_token_count(p) <= TIER2_MAX_TOKENS
    ]
    delete_points(collection, targets, dry_run, "Tier 2 — low-token micro raw-chunks (confidence > 0)")
    return set(targets)


def tier3_hash_dedup(points: list, dry_run: bool, collection: str) -> set:
    """Exact source_hash duplicates — keep highest confidence, delete the rest."""
    hash_map: dict[str, list] = defaultdict(list)
    for p in points:
        h = get_source_hash(p)
        if h:
            hash_map[h].append(p)

    to_delete = []
    dupe_groups = 0
    for h, group in hash_map.items():
        if len(group) < 2:
            continue
        dupe_groups += 1
        group.sort(key=get_confidence, reverse=True)
        to_delete.extend(p.id for p in group[1:])

    print(f"  Found {dupe_groups:,} duplicate hash groups")
    delete_points(collection, to_delete, dry_run, "Tier 3 — exact source_hash duplicates")
    return set(to_delete)


# ------------------------------------------------------------------ stats ---

def print_stats(points: list, source_filter: str | None) -> None:
    total = len(points)
    zero_conf = sum(1 for p in points if get_confidence(p) == 0.0)
    micro = sum(1 for p in points if is_raw_chunk(p) and get_token_count(p) <= TIER1_MAX_TOKENS)
    micro_zero = sum(1 for p in points if is_raw_chunk(p) and get_confidence(p) == 0.0 and get_token_count(p) <= TIER1_MAX_TOKENS)
    micro_some = sum(1 for p in points if is_raw_chunk(p) and get_confidence(p) > 0.0 and get_token_count(p) <= TIER2_MAX_TOKENS)

    # Count duplicate hashes
    hash_map: dict[str, list] = defaultdict(list)
    for p in points:
        h = get_source_hash(p)
        if h:
            hash_map[h].append(p)
    dupe_ids = sum(len(g) - 1 for g in hash_map.values() if len(g) > 1)
    dupe_groups = sum(1 for g in hash_map.values() if len(g) > 1)

    print(f"\n  Stats for: {source_filter or 'ALL sources'}")
    print(f"  {'─'*40}")
    print(f"  Total points          : {total:>8,}")
    print(f"  Zero confidence       : {zero_conf:>8,}")
    print(f"  Micro (≤{TIER1_MAX_TOKENS} tokens)     : {micro:>8,}")
    print(f"  Tier 1 targets        : {micro_zero:>8,}  (zero-conf + micro)")
    print(f"  Tier 2 targets        : {micro_some:>8,}  (micro, confidence > 0)")
    print(f"  Tier 3 dupe groups    : {dupe_groups:>8,}  ({dupe_ids:,} extra copies)")
    print(f"  {'─'*40}")
    print(f"  Max deletable         : {micro_zero + micro_some + dupe_ids:>8,}")
    print()


# ------------------------------------------------------------------ main ----

def run(
    collection: str,
    source: str | None,
    tiers: set,
    dry_run: bool,
    stats_only: bool,
) -> None:
    label = source or "ALL sources"
    print(f"\n{'='*60}")
    print(f"  Corpus Scrubber — {collection}")
    print(f"  Source filter : {label}")
    if not stats_only:
        print(f"  Tiers         : {sorted(tiers)}")
        print(f"  Mode          : {'DRY RUN (no changes)' if dry_run else '⚠  LIVE — changes will be written'}")
    print(f"{'='*60}\n")

    print(f"Scrolling {label} from {collection}...")
    points = scroll_all(collection, source)
    print(f"Loaded {len(points):,} points")

    if not points:
        print("Nothing to process.")
        return

    print_stats(points, source)

    if stats_only:
        return

    deleted_t1: set = set()
    deleted_t2: set = set()
    deleted_t3: set = set()

    if 1 in tiers:
        deleted_t1 = tier1_hard_delete(points, dry_run, collection)

    if 2 in tiers:
        deleted_t2 = tier2_soft_delete(points, deleted_t1, dry_run, collection)

    if 3 in tiers:
        deleted_t3 = tier3_hash_dedup(points, dry_run, collection)

    total_deleted = len(deleted_t1) + len(deleted_t2) + len(deleted_t3)
    print(f"\n{'─'*60}")
    print(f"{'[DRY RUN] Total that WOULD BE deleted' if dry_run else 'Total deleted'}: {total_deleted:,}")
    print(f"Estimated remaining: {len(points) - total_deleted:,} / {len(points):,}")

    if dry_run:
        print("\nRe-run without --dry-run to apply changes.")
    else:
        print("\n✓ Done. Re-run Vectoreologist + canon_alignment_report to verify topology.")


def main():
    parser = argparse.ArgumentParser(
        description="Corpus-level dedup / noise scrubber for Qdrant collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python corpus_scrubber.py --stats --source thetanakh
  python corpus_scrubber.py --dry-run --source thetanakh --all-tiers
  python corpus_scrubber.py --source thetanakh --tier 1
  python corpus_scrubber.py --source thetanakh --all-tiers
  python corpus_scrubber.py --all-sources --all-tiers
        """
    )
    parser.add_argument("--collection",  default=DEFAULT_COLLECTION, help="Qdrant collection name")
    parser.add_argument("--source",      default=None,               help="Filter by source_file (e.g. thetanakh)")
    parser.add_argument("--all-sources", action="store_true",        help="Run across all sources (no filter)")
    parser.add_argument("--tier",        type=int, action="append",  dest="tiers", metavar="N",
                        help="Tier(s) to run: 1, 2, 3 (repeatable)")
    parser.add_argument("--all-tiers",   action="store_true",        help="Run all three tiers")
    parser.add_argument("--dry-run",     action="store_true",        help="Preview only — no deletes")
    parser.add_argument("--stats",       action="store_true",        help="Print stats only, no action")
    args = parser.parse_args()

    stats_only = args.stats

    # Resolve tiers (not needed for --stats)
    if not stats_only:
        if args.all_tiers:
            tiers = {1, 2, 3}
        elif args.tiers:
            tiers = set(args.tiers)
        else:
            print("Error: specify --tier N, --all-tiers, or --stats")
            parser.print_help()
            sys.exit(1)
    else:
        tiers = set()

    # Resolve source
    source = None if args.all_sources else args.source

    # Safety gate — live run with no source filter requires explicit confirmation
    if not args.dry_run and not stats_only and source is None:
        confirm = input(
            "\n⚠  WARNING: Live run with no source filter will scrub ALL sources in the collection.\n"
            "Type 'yes I know what I am doing' to continue: "
        )
        if confirm.strip() != "yes I know what I am doing":
            print("Aborted.")
            sys.exit(0)

    run(
        collection=args.collection,
        source=source,
        tiers=tiers,
        dry_run=args.dry_run,
        stats_only=stats_only,
    )


if __name__ == "__main__":
    main()
