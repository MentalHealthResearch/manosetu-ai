# Benchmark Response Generation

This folder generates response CSVs for the four unfinished models and both conditions: 40 questions x 4 models x 2 conditions = 320 responses. Qwen3-32B is deliberately excluded because it was already completed.

The benchmark model labels are `Llama-3.1-8B`, `Mistral-7B`, `Gemma-2-9B`, and `MentalLLaMA-13B`. The listed task describes four remaining models, not five.

## Before running

Install and start Ollama. Pull the exact model tags in `benchmark_config.json`; update only a tag in that new config file if your local MentalLLaMA tag differs.

```powershell
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull gemma2:9b
ollama pull mental-llama:13b
```

With Qdrant running and `narrative_therapy_corpus` present, generate all 320 responses:

```powershell
& 'C:\Users\goura\anaconda3\envs\narrative-rag\python.exe' run_benchmark.py
```

The eight response files are written to `../generated_responses/`. Every RAG output retains retrieved source IDs and similarity scores, allowing retrieval and response quality to be audited.

## Human evaluation

The five files in `../rater_packets/` remain for Psychologist 1 through Psychologist 5. Only the psychologists should enter values in those files. Generated model responses can be inserted into the existing response manifest/workbook only after generation is complete.
