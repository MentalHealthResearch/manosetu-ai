# Manosetu AI

Code and results for our study (MythoClinic in the paper) on using stories from Indian epics (the Ramayana, the Mahabharata and the Bhagavad Gita) in mental-health support chatbots.

We annotated 735 episodes from these texts with situations, teachings and psychological themes, and mapped the themes to 13 constructs such as stress, anxiety, loneliness and self-esteem. We then asked five open models 40 everyday questions about distress, once with retrieved episodes (RAG) and once without (No-RAG), and compared the answers.

## What is in this repository

| Folder | Contents |
|---|---|
| `psychologist_rag_evaluation_kit/benchmark/` | RAG1 and No-RAG generation for the five models (`run_benchmark.py`, `run_qwen3.py`) |
| `psychologist_rag_evaluation_kit/data/questions.csv` | The 40 benchmark questions |
| `psychologist_rag_evaluation_kit/retrieval_ground_truth/` | Construct label for 30 of the questions, used to score retrieval |
| `psychologist_rag_evaluation_kit/ai_agent_evaluation/` | Instructions given to the LLM judge (rubric, five evaluator personas, Cultural Alignment Score) |
| `psychologist_rag_evaluation_kit/conversations/` | Multi-turn example conversations (Gemma-2-9B, RAG and No-RAG) |
| `psychologist_rag_evaluation_kit/models/` | Ollama Modelfile for MentaLLaMA-13B |
| `mental_health_research/scripts/` | Builds the Qdrant index used by RAG1 |
| `rag_v2/` | RAG2: construct-aware hybrid retrieval, retrieval ablation (A0–A5), generation, statistics |
| `results/` | Generated responses, judge ratings, CAS scores, grounding audit and retrieval results |
| `docs/prompt_supplement.txt` | Exact prompts for every condition and every evaluation step |

The episode corpus itself is not included in this repository.

## Systems compared

- **No-RAG**: the model answers without any story.
- **RAG1**: the top 3 episodes by cosine similarity (all-mpnet-base-v2, Qdrant) are added to the prompt.
- **RAG2** (Gemma-2-9B only): the question is first analysed for constructs and risk, then one episode is picked by dense, HyDE and BM25 search with rank fusion and a construct boost. The answer is checked against the episode, and a safety instruction is added for risky messages.
- **Memory control** (Gemma-2-9B only): the model may tell a story from its own knowledge, without retrieval.

Models: Llama-3.1-8B, Mistral-7B, Gemma-2-9B, Qwen3-8B and MentaLLaMA-chat-13B, all run locally with Ollama at temperature 0.

## How the answers were evaluated

- **Response quality**: an LLM judge (Claude) scored every answer from 1 to 5 on empathy, relevance, clarity, actionability, justification and helpfulness, plus a safety flag. The judge was run as five evaluator personas. These are personas of one model, not independent human raters.
- **Cultural Alignment Score (CAS)**: four items (familiarity, situational relevance, cultural authenticity, motivational impact), scored by the same judge from the user's point of view.
- **Grounding audit**: for RAG1 and RAG2, whether the story told matches the episode the model was given.
- **Retrieval**: whether the retrieved episodes carry the question's construct label (top-1, top-3, MRR@10). This is a match against the annotations, not an expert judgement of the episode.

Answers were blinded (model and condition hidden) during scoring. Ratings by psychologists are being collected separately and are not part of this repository.

## Result files

- `results/responses/`: one CSV per model and condition. `RAG` is RAG1, `RAG-v2` is RAG2 and `NoRAG-narrative` is the memory control.
- `results/judge_ratings.csv`: one row per persona per answer.
- `results/cas_scores.csv`: one row per answer.
- `results/grounding_audit_rag1_vs_rag2.csv`
- `results/retrieval/`: per-question retrieval scores and the A0–A5 summary.

Some answers were scored again in later runs to check that the judge stayed consistent. These rows have `anchor = 1` and are left out of the analysis.

## Running

Requirements: Python 3.11, Ollama, and Docker (for Qdrant, RAG1 only).

```
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull gemma2:9b
ollama pull qwen3:8b
```

MentaLLaMA-13B is not in the Ollama library. Download the Q2_K GGUF of `klyang/MentaLLaMA-chat-13B` (TensorBlock), save it as `psychologist_rag_evaluation_kit/models/MODEL.gguf`, then run:

```
cd psychologist_rag_evaluation_kit/models
ollama create mental-llama:13b -f Modelfile.mentalllama
```

The scripts expect the corpus in `mental_health_research/data/dataset/` (folders `Ramayanam`, `Gita` and `Mahabharat`).

```
# RAG1 index
docker run -d -p 6333:6333 qdrant/qdrant
cd mental_health_research
python scripts/ingest_all_narrative_corpus.py

# RAG1 and No-RAG answers
cd ../psychologist_rag_evaluation_kit/benchmark
python run_benchmark.py
python run_qwen3.py

# RAG2: retrieval ablation, generation, memory control
cd ../../rag_v2
python eval_retrieval.py
python generate_v2.py --prompt-version 2.1
python generate_v2.py --prompt-version 2.1 --parametric

# statistics reported in the paper (from the repository root)
python rag_v2/mythoclinic_stats.py
```

`--prompt-version 2.1` is the version used for the reported results. Version 2.2, the default, adds extra safety triggers that were introduced after the evaluation.

## Note

This is a research prototype. It is not a clinical tool and should not be used to support people in distress without professional oversight. In India, Tele-MANAS can be reached at 14416.
