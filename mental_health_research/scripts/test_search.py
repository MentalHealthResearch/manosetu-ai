#!/usr/bin/env python3
"""
test_search.py — Verify Qdrant is populated and returning relevant results.

Adapted from MythoClinic_v3/scripts/test_search.py
Run after ingest_ramayana.py to sanity-check retrieval quality.
"""

import os
import sys

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ramayana_stories")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 3

# ---------------------------------------------------------------------------
# Sample mental health queries (same style as MythoClinic test_search.py)
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = [
    "I feel so angry all the time and I can't control my rage",
    "I'm afraid of losing the people I love",
    "I feel like giving up — nothing matters anymore",
    "I keep worrying about everything and can't relax",
    "I feel isolated and lonely even when surrounded by people",
]


def main() -> None:
    print("=" * 70)
    print("Qdrant Retrieval Test — Ramayana Stories")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"\nLoading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ------------------------------------------------------------------
    # 2. Connect to Qdrant
    # ------------------------------------------------------------------
    print(f"Connecting to Qdrant at {QDRANT_URL} ...")
    client = QdrantClient(url=QDRANT_URL)

    # Check collection exists
    collections = client.get_collections()
    names = {c.name for c in collections.collections}
    if QDRANT_COLLECTION not in names:
        print(f"ERROR: Collection '{QDRANT_COLLECTION}' not found.", file=sys.stderr)
        print(f"Available collections: {names}", file=sys.stderr)
        sys.exit(1)

    count = client.count(collection_name=QDRANT_COLLECTION)
    print(f"Collection: {QDRANT_COLLECTION}  |  Points: {count.count}")

    # ------------------------------------------------------------------
    # 3. Run sample queries
    # ------------------------------------------------------------------
    print(f"\nRunning {len(SAMPLE_QUERIES)} sample queries (top_k={TOP_K})\n")

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"{'─' * 70}")
        print(f"QUERY {i}: {query}")
        print(f"{'─' * 70}")

        # Embed
        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        # Search (qdrant-client 1.19+ uses query_points, not search)
        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            limit=TOP_K,
            with_payload=True,
        ).points

        if not results:
            print("  [No results returned]\n")
            continue

        for rank, scored_point in enumerate(results, 1):
            payload = scored_point.payload or {}
            story_id = payload.get("story_id", "N/A")
            kanda = payload.get("kanda", "N/A")
            themes = payload.get("themes", [])
            paragraph = payload.get("paragraph", "")[:200]
            chapter = payload.get("chapter", "")

            print(f"  [{rank}] score={scored_point.score:.4f} | {story_id} | {kanda}")
            if chapter:
                print(f"       Chapter: {chapter}")
            if themes:
                print(f"       Themes: {', '.join(str(t) for t in themes)}")
            if paragraph:
                print(f"       Excerpt: {paragraph}...")
            print()

    print("=" * 70)
    print("Test complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
