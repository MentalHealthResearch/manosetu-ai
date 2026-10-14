# RAG-v2: construct-aware hybrid retrieval with reranking and grounded generation

Runs **alongside** the existing pipeline. It only *reads* `mental_health_research/data/dataset`, the kit's
`questions.csv` and the ground-truth template; it writes only to `rag_v2/outputs/`. No existing file or Qdrant
collection is touched.

## Pipeline (per user message)

| Stage | Component | Why (limitation of the current "vanilla" RAG it fixes) |
|---|---|---|
| 1 | **Query understanding** (Gemma-2-9B, JSON): 13-construct detection, risk level, one-line situation, HyDE-style *hypothetical episode* | Queries and episodes use different vocabularies; top-1 cosine averaged only 0.27 |
| 2 | **Hybrid retrieval**: bge-m3 dense (raw query + hypothetical episode) + BM25 over annotated fields, fused with Reciprocal Rank Fusion | Single mpnet vector; no lexical signal; English-only encoder |
| 3 | **Construct boost** using the corpus `Mental_Health_Themes` annotations | Retrieval ignored the construct annotations (e.g. Dasaratha episode for a family-conflict query) |
| 4 | **Cross-encoder rerank** (bge-reranker-v2-m3) of top-20 + **abstention threshold** | Top-3 always injected even when irrelevant; 4 episodes dominated 40 queries |
| 5 | **Grounded, construct-aware prompt** with ONE structured episode; explicit no-invention rules | Models cited non-retrieved characters, invented verse numbers, "Passage 2 says…" meta-talk |
| 6 | **Faithfulness verifier** (LLM-as-checker) → one regeneration if unsupported claims | No check of story faithfulness existed |
| 7 | **Safety gate** (LLM risk + regex) → explicit safety check-in + Tele-MANAS 14416 | "I should just give up" got no safety check in either condition |

## Experiments

1. `eval_retrieval.py` — ablation ladder **A0** (current: mpnet dense) → **A1** bge-m3 → **A2** +query enrichment →
   **A3** +BM25 hybrid → **A4** +construct boost → **A5** +rerank. Metrics: construct-precision@1/@3, construct-MRR@10
   on the 30 labelled queries, diversity (distinct episodes across 40 queries), and — once experts fill
   `expected_episode_id` in the ground-truth file — episode-level **Recall@3 / MRR** (computed automatically).
2. `generate_v2.py` — 40 answers with the full pipeline (`RAG-v2`), same CSV schema as the benchmark plus audit
   columns (detected construct, risk, used episode, abstained, faithfulness before/after regeneration).
3. `generate_v2.py --parametric` — **control arm** `NoRAG-narrative`: identical prompt but the model may use stories
   from its own memory. Comparing RAG-v2 vs this arm isolates the value of retrieval (grounding/faithfulness), which
   the original No-RAG arm (stories forbidden) could not.

All outputs can be scored unchanged by the existing 5-agent judges, the CAS scorer and the psychologist packets.

## Run

Double-click `RUN_RAG_V2.bat` (Ollama with `gemma2:9b` running; first run downloads ~4.5 GB of HF models).
`--mock` on any script runs a model-free smoke test (numbers meaningless).

## Caveats for the paper
- C-metrics use episode construct annotations as *silver* labels; A4/A5 use the detected construct as a boost, so
  report construct-detection accuracy next to them and treat expert episode ground truth as the decisive test.
- Faithfulness is judged by an LLM (Gemma) against the episode fields; spot-check a sample manually.
- Corpus coverage limits any retriever: only 18 episodes are tagged Work-Life Balance and 41 Loneliness.
