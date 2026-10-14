# RAG / No-RAG Psychologist Evaluation Kit

This folder is a self-contained evaluation workflow for the four models that still need evaluation:

- Llama-3.1-8B
- Mistral-7B
- Gemma-2-9B
- MentalLLaMA-13B

Each model is evaluated in `RAG` and `No-RAG` conditions on the same 40 primary questions. Five psychologists independently rate every answer on Empathy, Relevance, Clarity, Actionability, Justification, and Helpfulness using a 1-5 scale.

## Files

- `outputs/f82004b3/LLM_RAG_NoRAG_psychologist_evaluation.xlsx` is the researcher workbook. Its `Table 5` sheet calculates the final means only after all required ratings are present.
- `data/questions.csv` contains the 40 primary prompts, matching the existing psychologist-feedback file.
- `data/response_manifest.csv` contains one response slot for every model, condition, and question (320 total). Add generated answers and run metadata here.
- `rater_packets/` contains five blank packet templates. Do not send these until responses have been generated and blinded.
- `scripts/build_rater_packets.py` creates randomized, blinded packets from completed responses and writes the private answer key.
- `scripts/validate_and_compute_table5.py` validates the five completed packets and produces `table5_means.csv` and `table5_means.md`.
- `protocol.md` contains the study procedure and the scoring rubric.

## Workflow

1. Generate 40 answers for each of the 10 model-condition combinations, keeping the question set, system prompt, generation settings, and output constraints fixed. The only intended difference is whether retrieval context is provided.
2. Put every answer and its run metadata into `data/response_manifest.csv`.
3. Run `python scripts/build_rater_packets.py`. It refuses to make packets until all 320 answers are present and creates a private answer key.
4. Give one blinded CSV packet to each psychologist. Each psychologist scores all six criteria for every answer without seeing model or condition.
5. Save completed files as `rater_packets/completed/psychologist_01.csv` through `psychologist_05.csv` and run `python scripts/validate_and_compute_table5.py`.
6. Use the generated means in the manuscript table. The script refuses to calculate a model-condition mean unless it has exactly 200 valid scores for that metric: 40 questions x 5 psychologists.

The supplied screenshot is used only as the target table structure. This kit does not invent scores for any uncollected model-condition rows.
