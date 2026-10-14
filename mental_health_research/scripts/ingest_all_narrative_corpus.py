#!/usr/bin/env python3
"""Ingest all Ramayanam, Gita, and Mahabharat JSON passages into Qdrant.

This intentionally leaves ingest_ramayana.py unchanged. Use this script for
the study-wide RAG knowledge base that contains all three narrative corpora.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_ROOT = Path("data/dataset")
DEFAULT_COLLECTION = "narrative_therapy_corpus"
DEFAULT_QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768
CORPORA = ("Ramayanam", "Gita", "Mahabharat")


def compact(value: Any) -> str:
    """Turn structured annotations into retrieval text without losing labels."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(compact(item) for item in value if compact(item))
    if isinstance(value, dict):
        return "; ".join(f"{key}: {compact(item)}" for key, item in value.items() if compact(item))
    return str(value)


def corpus_name(path: Path, root: Path) -> str:
    first_part = path.relative_to(root).parts[0]
    return first_part.lower()


def read_json_objects(json_file: Path) -> list[dict[str, Any]]:
    """Read a normal JSON object or a comma-separated stream of JSON objects."""
    text = json_file.read_text(encoding="utf-8-sig").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        if position >= len(text):
            break
        try:
            parsed, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON object in {json_file}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object in {json_file}, got {type(parsed).__name__}.")
        records.append(parsed)
    if not records:
        raise ValueError(f"No JSON objects found in {json_file}.")
    return records


def load_records(dataset_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for corpus in CORPORA:
        corpus_root = dataset_root / corpus
        if not corpus_root.exists():
            print(f"Warning: corpus not found: {corpus_root}")
            continue
        for json_file in sorted(corpus_root.rglob("*.json")):
            try:
                raw_records = read_json_objects(json_file)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Could not read {json_file}: {exc}") from exc
            relative_id = json_file.relative_to(corpus_root).with_suffix("").as_posix()
            source = corpus_name(json_file, dataset_root)
            for part_index, raw in enumerate(raw_records, start=1):
                suffix = "" if len(raw_records) == 1 else f"#part{part_index}"
                records.append(
                    {
                        "story_id": f"{source}:{relative_id}{suffix}",
                        "corpus": source,
                        "relative_path": json_file.relative_to(dataset_root).as_posix(),
                        "paragraph": raw.get("Paragraph", raw.get("paragraph", "")),
                        "source_text": raw.get("Source_Text", ""),
                        "chapter": raw.get("Chapter/Canto/Verse_Reference", ""),
                        "characters": raw.get("Characters_Involved", []),
                        "situation": raw.get("Situation/Problem_Faced", ""),
                        "key_teaching": raw.get("Key_Teaching/Action_Taken", ""),
                        "outcome": raw.get("Outcome/Resolution", ""),
                        "themes": raw.get("Mental_Health_Themes", []),
                        "theme_synonyms": raw.get("Mental_Health_Theme_Synonyms", {}),
                        "psych_summary": raw.get("Brief_Summary_Psychological", ""),
                        "spiritual_summary": raw.get("Brief_Summary_Spiritual", ""),
                        "concepts": raw.get("Spiritual_Psychological_Concept_Indian", []),
                    }
                )
    return records


def retrieval_text(record: dict[str, Any]) -> str:
    fields = (
        ("Corpus", record["corpus"]),
        ("Source", record["source_text"]),
        ("Reference", record["chapter"]),
        ("Characters", record["characters"]),
        ("Situation", record["situation"]),
        ("Teaching", record["key_teaching"]),
        ("Outcome", record["outcome"]),
        ("Mental health themes", record["themes"]),
        ("Theme synonyms", record["theme_synonyms"]),
        ("Psychological summary", record["psych_summary"]),
        ("Spiritual summary", record["spiritual_summary"]),
        ("Concepts", record["concepts"]),
        ("Passage", record["paragraph"][:1000]),
    )
    return "\n".join(f"{label}: {compact(value)}" for label, value in fields if compact(value))


def prepare_collection(client: Any, name: str, recreate: bool) -> None:
    from qdrant_client.models import Distance, VectorParams

    existing = {item.name for item in client.get_collections().collections}
    if recreate and name in existing:
        client.delete_collection(collection_name=name)
        existing.remove(name)
        print(f"Deleted collection: {name}")
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest all narrative therapy corpora into Qdrant")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recreate", action="store_true", help="Replace only the named all-corpus collection")
    parser.add_argument("--validate-only", action="store_true", help="Read and validate JSON without embedding or uploading")
    args = parser.parse_args()

    records = load_records(Path(args.dataset_root))
    if not records:
        raise SystemExit("No JSON narrative records found.")
    counts: dict[str, int] = {}
    for record in records:
        counts[record["corpus"]] = counts.get(record["corpus"], 0) + 1
    print(f"Loaded {len(records)} records: {counts}")
    if len({record["story_id"] for record in records}) != len(records):
        raise SystemExit("Duplicate story IDs detected.")
    if args.validate_only:
        print("Validation passed. No embeddings or Qdrant changes were made.")
        return

    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors: np.ndarray = model.encode(
        [retrieval_text(record) for record in records],
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    client = QdrantClient(url=args.qdrant_url)
    prepare_collection(client, args.collection, args.recreate)
    points = [
        PointStruct(id=index, vector=vectors[index].tolist(), payload=record)
        for index, record in enumerate(records)
    ]
    for start in range(0, len(points), args.batch_size):
        client.upsert(collection_name=args.collection, points=points[start : start + args.batch_size], wait=True)
    print(f"Uploaded {len(points)} records to '{args.collection}'.")


if __name__ == "__main__":
    main()
