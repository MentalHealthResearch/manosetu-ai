#!/usr/bin/env python3
"""
generate_responses_rag.py — RAG-backed response generator (STRICT PROMPT).

For each question:
  1. Embed the question
  2. Retrieve top-K relevant mythological passages from Qdrant
  3. Inject FULL passages into a STRICT prompt that FORCES citation
  4. Generate response via ollama.chat()

The strict prompt uses negative prompting to prohibit parametric knowledge
and requires the model to cite exact Story IDs from the retrieved passages.

Output: data/generated_responses_rag.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_INPUT_CSV = "test_questions.csv"
DEFAULT_OUTPUT_CSV = "data/generated_responses_rag.csv"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "ramayana_stories"
DEFAULT_TOP_K = 3

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------

def _check_model_available(model_name: str) -> None:
    """Fail fast if the requested Ollama model is not pulled."""
    try:
        models = ollama.list()
        # Handle both dict-style and object-style responses
        model_list = models.get("models", []) if hasattr(models, "get") else getattr(models, "models", [])
        available = set()
        for m in model_list:
            name = m.get("model", "") if isinstance(m, dict) else getattr(m, "model", "")
            available.add(name)
        if model_name not in available:
            # Try matching without tag if needed
            stripped = model_name.split(":")[0]
            for avail in available:
                if avail == model_name or avail.startswith(stripped):
                    return
            print(
                f"ERROR: Model '{model_name}' not found in Ollama. "
                f"Available: {sorted(available)}. Run: ollama pull {model_name}",
                file=sys.stderr,
            )
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama. Is it running? {e}", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------

def retrieve_stories(
    client: QdrantClient,
    model: SentenceTransformer,
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Embed query_text and search Qdrant for top-K relevant story passages.

    Returns list of dicts with keys:
      - story_id
      - kanda
      - chapter
      - themes
      - paragraph
      - psych_summary
      - spiritual_summary
      - key_teaching
      - score (cosine similarity)
    """
    query_embedding = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    results = client.query_points(
        collection_name=DEFAULT_COLLECTION,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    stories = []
    for scored_point in results:
        payload = scored_point.payload or {}
        stories.append({
            "story_id": payload.get("story_id", "N/A"),
            "kanda": payload.get("kanda", "N/A"),
            "chapter": payload.get("chapter", ""),
            "themes": payload.get("themes", []),
            "paragraph": payload.get("paragraph", ""),
            "psych_summary": payload.get("psych_summary", ""),
            "spiritual_summary": payload.get("spiritual_summary", ""),
            "key_teaching": payload.get("key_teaching", ""),
            "score": scored_point.score,
        })

    return stories


# ---------------------------------------------------------------------------
# Prompt builder (strict — forces citation of retrieved passages)
# ---------------------------------------------------------------------------

def build_rag_prompt(query_text: str, retrieved_stories: list[dict]) -> str:
    """
    Build a strict prompt that FORCES the model to use retrieved passages
    and explicitly forbids using parametric knowledge.
    """
    # Build context block with full paragraphs (not truncated)
    context_parts = []
    for i, story in enumerate(retrieved_stories, 1):
        context_parts.append(f"[STORY {i} — ID: {story['story_id']}]")
        context_parts.append(f"Kanda: {story['kanda']}")
        if story["chapter"]:
            context_parts.append(f"Chapter: {story['chapter']}")
        if story["themes"]:
            context_parts.append(f"Themes: {', '.join(str(t) for t in story['themes'])}")
        if story["psych_summary"]:
            context_parts.append(f"Psychological Summary: {story['psych_summary']}")
        if story["key_teaching"]:
            context_parts.append(f"Key Teaching: {story['key_teaching']}")
        if story["paragraph"]:
            context_parts.append(f"Full Text: {story['paragraph']}")
        context_parts.append("")

    context_block = "\n".join(context_parts)

    story_id_list = ", ".join(s["story_id"] for s in retrieved_stories)

    prompt = f"""You are a mental health counselor chatbot. Below are EXACTLY THREE mythological passages retrieved from our database. You MUST use ONLY these passages. You are FORBIDDEN from using any prior knowledge of Indian mythology.

{context_block}

---
STRICT INSTRUCTIONS — FOLLOW THESE EXACTLY:

1. BEFORE writing your response, list the Story IDs you are going to use: {story_id_list}
2. Choose EXACTLY ONE story from the three passages above.
3. Summarize that chosen story in your own words, using ONLY details present in its Full Text.
4. Connect that story to the user's question with a clear, specific parallel.
5. End with 2-3 actionable suggestions.
6. Your response MUST mention the chosen story by its exact Story ID at least once.
7. Your response MUST be under 250 words.

---
FORBIDDEN — DO NOT DO ANY OF THESE:
- Do NOT use any mythological story not listed above.
- Do NOT mention characters or events not found in the passages above.
- Do NOT use generic references like "In our tradition..." or "Ancient texts say..."
- Do NOT draw from your training data about Ramayana or Mahabharata.
- Do NOT fabricate details about the story.
- If none of the three stories above are relevant to the user's question, say so explicitly and give general advice WITHOUT any mythological reference.

---
User Question: {query_text}

Now, write your response following ALL instructions above."""

    return prompt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAG-backed responses")
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Input questions CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Output responses CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant URL")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of passages to retrieve")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--delay", type=float, default=0.5, help="Rate limit delay between calls (seconds)")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N questions (for testing)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load questions
    # ------------------------------------------------------------------
    print(f"Loading questions from {args.input} ...")
    df = pd.read_csv(args.input)
    if args.limit > 0:
        df = df.head(args.limit)
        print(f"Limited to first {args.limit} rows")
    print(f"Total questions: {len(df)}")

    # ------------------------------------------------------------------
    # 2. Load embedding model + Qdrant
    # ------------------------------------------------------------------
    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to Qdrant at {args.qdrant_url} ...")
    qdrant = QdrantClient(url=args.qdrant_url)

    # Verify collection exists
    collections = qdrant.get_collections()
    names = {c.name for c in collections.collections}
    if args.collection not in names:
        print(f"ERROR: Collection '{args.collection}' not found.", file=sys.stderr)
        print(f"Available: {names}", file=sys.stderr)
        sys.exit(1)
    print(f"Collection '{args.collection}' ready.")

    # Verify model is available before starting expensive generation
    _check_model_available(args.model)
    print(f"Model '{args.model}' ready.")

    # ------------------------------------------------------------------
    # 3. Generate responses
    # ------------------------------------------------------------------
    results = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating RAG-backed responses using {args.model} (top_k={args.top_k}) ...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        qid = row.get("question_id", "")
        sqid = row.get("subquestion_id", "")
        q_text = row.get("question_text", "")

        if not q_text:
            print(f"Warning: empty question_text for qid={qid}, sqid={sqid}", file=sys.stderr)
            continue

        # --- Retrieve ---
        retrieved = retrieve_stories(qdrant, embed_model, q_text, top_k=args.top_k)
        story_ids = [s["story_id"] for s in retrieved]
        scores = [s["score"] for s in retrieved]

        # --- Build prompt ---
        prompt = build_rag_prompt(q_text, retrieved)

        # --- Generate ---
        success = False
        retries = 3
        delay = args.delay
        generated_text = ""
        tokens_used = 0
        generation_time = 0.0
        error_msg = ""

        while not success and retries > 0:
            try:
                start_time = time.time()
                response = ollama.chat(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": args.temperature},
                )
                generation_time = time.time() - start_time
                generated_text = response["message"]["content"]
                tokens_used = response.get("eval_count", 0)
                success = True
                time.sleep(delay)
            except Exception as e:
                retries -= 1
                error_msg = str(e)
                print(f"  Error qid={qid}, sqid={sqid}: {e}")
                if retries > 0:
                    backoff = 2 ** (3 - retries)
                    print(f"  Retrying in {backoff}s ... ({retries} left)")
                    time.sleep(backoff)
                else:
                    print(f"  Max retries reached. Skipping.")
                    generated_text = f"ERROR: {error_msg}"

        results.append({
            "question_id": qid,
            "subquestion_id": sqid,
            "question_text": q_text,
            "retrieved_story_ids": ";".join(story_ids),
            "retrieval_scores": ";".join(f"{s:.4f}" for s in scores),
            "prompt": prompt,
            "generated_response": generated_text,
            "tokens_used": tokens_used,
            "generation_time_seconds": generation_time,
        })

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(results_df)} responses to {args.output}")

    errors = results_df[results_df["generated_response"].str.startswith("ERROR:")]
    if len(errors) > 0:
        print(f"Warning: {len(errors)} responses failed to generate.")


if __name__ == "__main__":
    main()
