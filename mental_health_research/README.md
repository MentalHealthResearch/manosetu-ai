# Mental Health Research — RAG Evaluation Pipeline

This project evaluates a RAG-backed therapy chatbot that generates culturally-grounded responses using Indian mythological stories (Ramayana). The pipeline retrieves relevant passages, forces the model to cite them, and evaluates response quality against psychologist-rated references.

## What This Does

1. **Retrieve** — For each mental health question, embed it and find the top 3 most relevant Ramayana stories from Qdrant.
2. **Generate** — Feed those stories into a strict prompt that forces the LLM (`qwen3:8b`) to cite exact Story IDs and use only the retrieved passages.
3. **Evaluate** — Compare generated responses against psychologist-approved references using BERTScore and ROUGE-L.
4. **Correlate** — Check whether automated metrics predict psychologist ratings.

## Prerequisites

- **Python 3.11+**
- **Ollama** installed and running with `qwen3:8b` pulled
- **Docker** (for Qdrant)
- **Dataset** — 291 Ramayana story JSON files in a `Dataset_ramayanam/` folder with subfolders `balakanda/`, `ayodhyakanda/`, `aranyakanda/`

## Setup

### 1. Clone / navigate to project

```bash
cd mental_health_research
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Qdrant (Docker)

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### 5. Verify Ollama model

```bash
ollama list                   # should show qwen3:8b
ollama pull qwen3:8b         # if missing
```

### 6. Prepare dataset

Place your `Dataset_ramayanam/` folder at the project root or anywhere accessible. You will pass its path in the next step.

Expected structure:
```
Dataset_ramayanam/
├── balakanda/
│   ├── story_001.json
│   └── ...
├── ayodhyakanda/
│   └── ...
└── aranyakanda/
    └── ...
```

## Running the Pipeline

### Step 0 — Ingest stories into Qdrant (one-time)

```bash
python scripts/ingest_ramayana.py --dataset-root /path/to/Dataset_ramayanam
```

If your dataset is at `/home/jd/Documents/Mythoclinic_v3/Dataset_ramayanam` (the default), you can just run:

```bash
python scripts/ingest_ramayana.py
```

This creates a `ramayana_stories` collection with 768-dimensional embeddings.

### Step 1 — Full pipeline (all 120 questions)

```bash
python src/run_full_pipeline.py
```

This runs all four phases in sequence:
1. `generate_responses_rag.py` — generate responses (~30–35 minutes)
2. `build_references_from_psychologists.py` — build reference table
3. `compute_metrics_rag.py` — compute BERTScore + ROUGE-L
4. `correlation_analysis_rag.py` — correlation report

### Step 2 — Check outputs

Results appear in `data/`:

| File | Contents |
|------|----------|
| `generated_responses_rag.csv` | All 120 generated responses with retrieved Story IDs |
| `psychologist_references.csv` | 37 psychologist-rated reference texts |
| `bertscore_rag.csv` | Per-subquestion BERTScore precision/recall/f1 |
| `rouge_rag.csv` | Per-subquestion ROUGE-L precision/recall/f1 |
| `metrics_summary_rag.csv` | Aggregated metrics merged with ratings |
| `correlation_report_rag.md` | Human-readable correlation report |

### Optional: Test with a subset

```bash
# Generate only first 5 questions for quick testing
python src/generate_responses_rag.py --limit 5 --output data/test_responses.csv
```

## Project Structure

```
mental_health_research/
├── src/
│   ├── generate_responses_rag.py          # RAG response generator
│   ├── build_references_from_psychologists.py  # Build reference table
│   ├── compute_metrics_rag.py             # BERTScore + ROUGE-L
│   ├── correlation_analysis_rag.py        # Correlation analysis
│   └── run_full_pipeline.py               # Orchestrator
├── scripts/
│   ├── ingest_ramayana.py                 # One-time data ingestion
│   └── test_search.py                     # Verify Qdrant retrieval
├── data/                                  # Generated outputs (empty initially)
├── archive/                               # Historical files
│   └── test_questions_answers.csv
├── test_questions.csv                     # 120 subquestions (28 themes)
├── Final_human_evaluations_improved.csv   # 5 psychologist ratings
├── requirements.txt                       # Dependencies
└── README.md                              # This file
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Embedding model | `sentence-transformers/all-mpnet-base-v2` | 768-dim, cosine similarity |
| LLM | `qwen3:8b` | Local via Ollama |
| Top-K retrieval | 3 | Number of passages retrieved per question |
| Vector DB | Qdrant (localhost:6333) | Cosine distance |
| Temperature | 0.0 | Deterministic generation |
| Reference threshold | ≥ 3.5 | Mean psychologist rating to qualify as reference |

## How the Strict Prompt Works

The prompt forces the model to:
1. List the Story IDs it will use
2. Choose exactly ONE retrieved story
3. Summarize it using ONLY details from the retrieved text
4. Connect it to the user's question
5. Cite the exact Story ID in the response
6. Stay under 250 words

It also uses **negative prompting** (explicit "DO NOT" rules) to block parametric knowledge:
- Do NOT use stories not in the retrieved passages
- Do NOT mention characters not found above
- Do NOT use generic references like "In our tradition..."
- Do NOT draw from training data about Ramayana or Mahabharata

## Metrics Explained

**BERTScore (F1)** — Semantic similarity using contextual embeddings (roberta-large). Range: 0–1. Higher = more semantically similar to reference.

**ROUGE-L (F1)** — Surface-level overlap measuring longest common subsequence. Range: 0–1. Lower here (~0.15) because the model paraphrases rather than copies.

**Spearman Correlation** — Rank-order correlation between metric scores and psychologist ratings. The key question: do automated scores predict human judgments?

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Collection not found" | Run `python scripts/ingest_ramayana.py` first |
| "Model not found" | Run `ollama pull qwen3:8b` |
| "Cannot reach Ollama" | Start Ollama service (`ollama serve`) |
| "Module not found" | Activate `.venv` and run `pip install -r requirements.txt` |
| Slow generation | Normal. qwen3:8b on CPU takes ~15s per question. 120 questions ≈ 30 min. |
| BERTScore first run slow | Downloads roberta-large (~500MB). One-time only. |

## Notes

- **Dataset not included.** You must provide your own `Dataset_ramayanam/` folder with JSON story files.
- **References are LLM-generated + psychologist-reviewed**, not independently human-written. BERTScore is inflated by this; ROUGE-L is more honest.
- **Correlation is underpowered** — only 37 questions have references, and ratings are compressed (3.5–4.3 on a 5-point scale). The architecture and qualitative outputs are the real contributions.

## Citation / Acknowledgment

If this pipeline is useful for your research, please cite the underlying models:
- Embedding: `sentence-transformers/all-mpnet-base-v2`
- LLM: `qwen3:8b` (Qwen3, Alibaba Cloud)
- Metrics: BERTScore (Zhang et al., 2020), ROUGE (Lin, 2004)
