#!/usr/bin/env python3
"""Generate 40 responses per condition for the four unfinished benchmark models.

Outputs are intentionally separate from human-rater files.  This script does
not create, infer, or populate psychologist ratings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

try:
    import ollama
except ImportError as exc:
    raise SystemExit("Install Ollama Python support first: pip install ollama") from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "benchmark_config.json"
OUTPUT_DIR = ROOT.parent / "generated_responses"


def available_models() -> set[str]:
    listing = ollama.list()
    items = listing.get("models", []) if isinstance(listing, dict) else listing.models
    return {item.get("model", "") if isinstance(item, dict) else item.model for item in items}


def as_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value or "")


def retrieve(client: QdrantClient, encoder: SentenceTransformer, collection: str, question: str, top_k: int) -> list[dict[str, Any]]:
    vector = encoder.encode(question, normalize_embeddings=True, convert_to_numpy=True).tolist()
    result = client.query_points(collection_name=collection, query=vector, limit=top_k, with_payload=True)
    return [{**(point.payload or {}), "retrieval_score": point.score} for point in result.points]


def rag_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    context = []
    for number, passage in enumerate(passages, start=1):
        context.append(
            "\n".join(
                [
                    f"[Passage {number}; source ID: {as_text(passage.get('story_id'))}]",
                    f"Source: {as_text(passage.get('corpus'))}",
                    f"Text: {as_text(passage.get('paragraph'))}",
                    f"Psychological summary: {as_text(passage.get('psych_summary'))}",
                    f"Key teaching/action: {as_text(passage.get('key_teaching'))}",
                ]
            )
        )
    context_block = "\n\n".join(context)
    return f"""You are a supportive mental-health conversation assistant. Provide culturally sensitive, non-judgmental support.

Use only the supplied narrative passages when making a narrative reference. Do not invent story details. Validate the person's feelings, make a specific connection if relevant, and give 2-3 practical next steps. Do not diagnose. If there is immediate danger or self-harm risk, encourage local emergency or crisis support.

Retrieved narrative passages:
{context_block}

User question: {question}

Write a warm, clear response of at most 250 words."""


def no_rag_prompt(question: str) -> str:
    return f"""You are a supportive mental-health conversation assistant. Provide culturally sensitive, non-judgmental support. Validate the person's feelings and give 2-3 practical next steps. Do not diagnose, do not invent narrative references, and do not claim access to a knowledge base. If there is immediate danger or self-harm risk, encourage local emergency or crisis support.

User question: {question}

Write a warm, clear response of at most 250 words."""


def generate(model_tag: str, prompt: str, temperature: float) -> tuple[str, int, float]:
    started = time.perf_counter()
    result = ollama.chat(model=model_tag, messages=[{"role": "user", "content": prompt}], options={"temperature": temperature})
    elapsed = time.perf_counter() - started
    return result["message"]["content"], int(result.get("eval_count", 0)), elapsed


def run_condition(model_label: str, model_tag: str, condition: str, questions: pd.DataFrame, config: dict[str, Any], client: QdrantClient | None, encoder: SentenceTransformer | None) -> None:
    output_path = OUTPUT_DIR / f"{model_label}__{condition}.csv"
    records: list[dict[str, Any]] = []
    done_ids: set[int] = set()
    partial_path = output_path.with_suffix(".partial.csv")
    if output_path.exists() and len(pd.read_csv(output_path)) == len(questions):
        print(f"Skipping {output_path.name}: already complete", flush=True)
        return
    if partial_path.exists():
        previous = pd.read_csv(partial_path)
        previous = previous[~previous["response_text"].astype(str).str.startswith("ERROR:")]
        records = previous.to_dict("records")
        done_ids = {int(value) for value in previous["question_id"]}
        print(f"Resuming {output_path.name}: {len(done_ids)} already done", flush=True)
    for _, row in questions.iterrows():
        if int(row["question_id"]) in done_ids:
            continue
        question = str(row["question_text"])
        passages: list[dict[str, Any]] = []
        if condition == "RAG":
            assert client is not None and encoder is not None
            passages = retrieve(client, encoder, config["qdrant_collection"], question, int(config["top_k"]))
            prompt = rag_prompt(question, passages)
        else:
            prompt = no_rag_prompt(question)
        response, tokens, elapsed = generate(model_tag, prompt, float(config["temperature"]))
        records.append({
            "answer_id": f"{model_label}__{condition}__Q{int(row['question_id']):02d}",
            "model": model_label,
            "model_tag": model_tag,
            "condition": condition,
            "question_id": int(row["question_id"]),
            "question_text": question,
            "response_text": response,
            "retrieved_story_ids": ";".join(as_text(passage.get("story_id")) for passage in passages),
            "retrieval_scores": ";".join(f"{passage['retrieval_score']:.6f}" for passage in passages),
            "rag_collection": config["qdrant_collection"] if condition == "RAG" else "",
            "prompt_version": "benchmark_v1",
            "generation_tokens": tokens,
            "generation_seconds": round(elapsed, 3),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        pd.DataFrame(records).to_csv(partial_path, index=False)
        print(f"{model_label} {condition}: question {int(row['question_id'])}/40 ({elapsed:.1f}s)", flush=True)
    frame = pd.DataFrame(records).sort_values("question_id")
    frame.to_csv(output_path, index=False)
    partial_path.unlink(missing_ok=True)
    print(f"Saved {len(records)} responses: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", action="append", help="Repeat to run only selected benchmark model labels")
    parser.add_argument("--condition", choices=["RAG", "No-RAG"], action="append")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    questions = pd.read_csv((args.config.parent / config["questions_csv"]).resolve())
    if len(questions) != 40:
        raise SystemExit(f"Expected exactly 40 questions, found {len(questions)}.")
    requested = set(args.model or config["models"].keys())
    unknown = requested.difference(config["models"])
    if unknown:
        raise SystemExit(f"Unknown model label(s): {sorted(unknown)}")
    installed = available_models()
    tags = {config["models"][label] for label in requested}
    missing = sorted(tag for tag in tags if tag not in installed)
    if missing:
        raise SystemExit("Missing Ollama model(s): " + ", ".join(missing) + ". Pull them, then rerun.")
    conditions = args.condition or ["RAG", "No-RAG"]
    OUTPUT_DIR.mkdir(exist_ok=True)
    client = encoder = None
    if "RAG" in conditions:
        client = QdrantClient(url=config["qdrant_url"])
        names = {item.name for item in client.get_collections().collections}
        if config["qdrant_collection"] not in names:
            raise SystemExit(f"Qdrant collection not found: {config['qdrant_collection']}")
        os.environ.setdefault("HF_HOME", str(ROOT.parent.parent / ".hf_cache"))
        encoder = SentenceTransformer(config["embedding_model"])
    for label, tag in config["models"].items():
        if label not in requested:
            continue
        for condition in conditions:
            run_condition(label, tag, condition, questions, config, client, encoder)


if __name__ == "__main__":
    main()
