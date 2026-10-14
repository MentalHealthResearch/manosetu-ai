# Retrieval Ground-Truth Set

`30_query_episode_ground_truth_template.csv` selects 30 distinct primary prompts from the project test set. It covers all 13 specified mental-health constructs:

- Stress Management: 3 pairs
- Anxiety: 3 pairs
- Fear: 2 pairs
- Low Mood: 2 pairs
- Sadness: 2 pairs
- Feelings of Hopelessness: 2 pairs
- Anger Management: 2 pairs
- Emotional Regulation: 2 pairs
- Relationship Challenges (Family | Romantic): 3 pairs
- Ego: 2 pairs
- Self-Esteem: 2 pairs
- Work-Life Balance: 2 pairs
- Loneliness: 3 pairs

Before retrieval evaluation, two domain reviewers should inspect the full narrative knowledge base and enter one or more valid `expected_episode_id` values, the matching episode title, and a short selection rationale for every pair. Valid episodes can come from Ramayanam, Gita, or Mahabharat. The expected episode must be chosen independently of the RAG system's output. Do not use the model's retrieved top result as its own ground truth.

After all rows are approved, copy the file to `30_query_episode_ground_truth.csv` and use it to calculate Recall@K and MRR from recorded retrieval results. The all-corpus ingester uses stable IDs in the form `ramayanam:<relative-path>`, `gita:<relative-path>`, and `mahabharat:<relative-path>`.
