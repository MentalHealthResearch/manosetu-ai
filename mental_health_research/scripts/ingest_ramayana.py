#!/usr/bin/env python3
"""
ingest_ramayana.py — Ingest Ramayana dataset into Qdrant

Adapted from MythoClinic_v3/scripts/ingest_qdrant.py.
Reads Dataset_ramayanam/*.json, embeds with all-mpnet-base-v2, upserts to Qdrant.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Default dataset location. Override with --dataset-root or set DATASET_ROOT env var.
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("DATASET_ROOT", "/home/jd/Documents/Mythoclinic_v3/Dataset_ramayanam")
)

# Kandas (subdirectories) to scan
KANDAS = ["balakanda", "ayodhyakanda", "aranyakanda"]

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768  # all-mpnet-base-v2 outputs 768-dim

# ---------------------------------------------------------------------------
# Qdrant config
# ---------------------------------------------------------------------------
DEFAULT_COLLECTION_NAME = "ramayana_stories"
DEFAULT_QDRANT_URL = "http://localhost:6333"

# ---------------------------------------------------------------------------
# Helper: build retrieval string from a JSON record
# ---------------------------------------------------------------------------

def build_retrieval_string(record: dict[str, Any]) -> str:
    """
    Build a dense retrieval string that merges all searchable text.

    Order matters: front-load the most semantically meaningful fields.
    """
    parts: list[str] = []

    # Mental health themes + synonyms (most important for matching queries)
    themes = record.get("Mental_Health_Themes", [])
    synonyms = record.get("Mental_Health_Theme_Synonyms", {})
    if themes:
        theme_strs = []
        for t in themes:
            syns = synonyms.get(t, "")
            if syns:
                theme_strs.append(f"{t}: {syns}")
            else:
                theme_strs.append(t)
        parts.append("Themes: " + "; ".join(theme_strs))

    # Situation / problem faced
    situation = record.get("Situation/Problem_Faced", "")
    if situation:
        parts.append(f"Situation: {situation}")

    # Key teaching / action taken
    teaching = record.get("Key_Teaching/Action_Taken", "")
    if teaching:
        parts.append(f"Teaching: {teaching}")

    # Psychological summary (detailed)
    psych_summary = record.get("Brief_Summary_Psychological", "")
    if psych_summary:
        parts.append(f"Psychological: {psych_summary}")

    # Spiritual / philosophical concepts
    concepts = record.get("Spiritual_Psychological_Concept_Indian", [])
    if concepts:
        concept_strs = []
        for c in concepts:
            if isinstance(c, dict):
                concept_strs.append(c.get("Concept", ""))
            elif isinstance(c, str):
                concept_strs.append(c)
        if concept_strs:
            parts.append("Concepts: " + ", ".join(filter(None, concept_strs)))

    # Outcome
    outcome = record.get("Outcome/Resolution", "")
    if outcome:
        parts.append(f"Outcome: {outcome}")

    # Paragraph (the raw story text) — appended last, truncated
    paragraph = record.get("Paragraph", "")
    if paragraph:
        # Truncate to avoid overly long retrieval strings
        parts.append(f"Passage: {paragraph[:500]}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

def load_stories(dataset_root: Path) -> list[dict[str, Any]]:
    """
    Walk dataset_root and load all .json files.
    Returns list of story dicts with additional fields:
      - story_id  (auto-generated)
      - kanda     (folder name)
      - file_name
    """
    stories: list[dict[str, Any]] = []
    for kanda in KANDAS:
        kanda_path = dataset_root / kanda
        if not kanda_path.exists():
            print(f"Warning: kanda path not found: {kanda_path}", file=sys.stderr)
            continue

        json_files = sorted(kanda_path.glob("*.json"))
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as fh:
                record = json.load(fh)

            # Normalize keys that use slashes
            record["story_id"] = f"{kanda}_{jf.stem}"
            record["kanda"] = kanda
            record["file_name"] = str(jf)
            stories.append(record)

    return stories


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = EMBEDDING_DIM,
    distance: Distance = Distance.COSINE,
    recreate: bool = False,
) -> None:
    """Create (or optionally recreate) a Qdrant collection."""
    if recreate:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection: {collection_name}")

    # Check if collection exists
    existing = client.get_collections()
    names = {c.name for c in existing.collections}
    if collection_name not in names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )
        print(f"Created collection: {collection_name} (dim={vector_size}, distance={distance.name})")
    else:
        print(f"Collection already exists: {collection_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Ramayana dataset into Qdrant")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant URL")
    parser.add_argument("--qdrant-api-key", default="", help="Qdrant API key (optional)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Path to Dataset_ramayanam folder")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection before ingest")
    parser.add_argument("--dry-run", action="store_true", help="Parse and embed only, do not upload to Qdrant")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--limit", type=int, default=0, help="Only ingest first N stories (for quick testing)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load stories
    # ------------------------------------------------------------------
    dataset_root = Path(args.dataset_root)
    print(f"Loading stories from {dataset_root} ...")
    stories = load_stories(dataset_root)
    if args.limit > 0:
        stories = stories[:args.limit]
        print(f"Limited to first {args.limit} stories")

    total = len(stories)
    print(f"Loaded {total} stories")

    if total == 0:
        print("No stories found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Build retrieval strings
    # ------------------------------------------------------------------
    print("Building retrieval strings ...")
    retrieval_strings = [build_retrieval_string(s) for s in stories]

    # Print a sample
    print(f"\nSample retrieval string (story_id={stories[0]['story_id']}):")
    print(retrieval_strings[0][:400] + "...\n")

    # ------------------------------------------------------------------
    # 3. Embed
    # ------------------------------------------------------------------
    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding retrieval strings ...")
    embeddings: np.ndarray = model.encode(
        retrieval_strings,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # ------------------------------------------------------------------
    # 4. Prepare Qdrant points
    # ------------------------------------------------------------------
    points: list[PointStruct] = []
    for idx, story in enumerate(stories):
        vector = embeddings[idx].tolist()

        # Build payload from story fields
        payload = {
            "story_id": story["story_id"],
            "kanda": story["kanda"],
            "file_name": story["file_name"],
            "paragraph": story.get("Paragraph", ""),
            "characters": story.get("Characters_Involved", []),
            "situation": story.get("Situation/Problem_Faced", ""),
            "key_teaching": story.get("Key_Teaching/Action_Taken", ""),
            "outcome": story.get("Outcome/Resolution", ""),
            "themes": story.get("Mental_Health_Themes", []),
            "theme_synonyms": story.get("Mental_Health_Theme_Synonyms", {}),
            "psych_summary": story.get("Brief_Summary_Psychological", ""),
            "spiritual_summary": story.get("Brief_Summary_Spiritual", ""),
            "concepts": [c.get("Concept", "") for c in story.get("Spiritual_Psychological_Concept_Indian", []) if isinstance(c, dict)],
            "source": story.get("Source_Text", ""),
            "chapter": story.get("Chapter/Canto/Verse_Reference", ""),
        }

        points.append(
            PointStruct(
                id=idx,  # use sequential integer IDs
                vector=vector,
                payload=payload,
            )
        )

    # ------------------------------------------------------------------
    # 5. Upsert to Qdrant (unless dry-run)
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\nDRY RUN — would upload the following:")
        print(f"  Points: {len(points)}")
        print(f"  Vector dim: {len(points[0].vector)}")
        print(f"  Sample payload keys: {list(points[0].payload.keys())}")
        sys.exit(0)

    print(f"\nConnecting to Qdrant at {args.qdrant_url} ...")
    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key or None)

    ensure_collection(
        client,
        args.collection,
        vector_size=EMBEDDING_DIM,
        distance=Distance.COSINE,
        recreate=args.recreate,
    )

    print(f"Upserting {len(points)} points into collection '{args.collection}' ...")
    client.upsert(
        collection_name=args.collection,
        points=points,
    )

    # Verify count
    count = client.count(collection_name=args.collection)
    print(f"\nDone. Collection '{args.collection}' now has {count.count} points.")


if __name__ == "__main__":
    main()
