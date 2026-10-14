# Evaluation Report -- Mental Health Research Project

## Pipeline Overview

This pipeline evaluates AI-generated mental health responses by comparing them against psychologist-approved references using two automated metrics: BERTScore and ROUGE-L.

**Phase 1 -- Generation:** The qwen3:8b model generated responses to all subquestions in test_questions.csv.

**Phase 2 -- Evaluation:** Each generated response was compared against the psychologist-rated reference for its question using:
- BERTScore F1 (roberta-large, layer 17) -- semantic similarity
- ROUGE-L F1 -- surface-level word overlap (longest common subsequence)

## Data Summary

| Metric | Value |
|--------|-------|
| Questions analyzed | 11 (all had psychologist rating >= 4.0) |
| Total subquestion pairs evaluated | 33 (3 subquestions per question) |
| Questions with psychologist mean rating >= 4.0 | 11 out of 19 evaluated |
| Psychologist mean rating range | 4.0 - 4.3 |
| Total generated responses | 899 subquestion responses across all 19 questions |

### Reference Selection

Questions were included as references if the mean of the 5 psychologists' overall_perceived_helpfulness rating was >= 4.0. All 11 qualifying questions had exactly 3 subquestions each, giving 33 candidate-reference pairs for metric computation.

## Results

### BERTScore (roberta-large, layer 17)

| Question ID | Psychologist Mean | BERTScore F1 |
|-------------|-----------------|--------------|
| 1 | 4.2 | 0.8465 |
| 3 | 4.3 | 0.8419 |
| 5 | 4.0 | 0.8481 |
| 11 | 4.1 | 0.8345 |
| 12 | 4.2 | 0.8534 |
| 13 | 4.2 | 0.8410 |
| 18 | 4.0 | 0.8417 |
| 19 | 4.0 | 0.8261 |
| 25 | 4.0 | 0.8598 |
| 34 | 4.2 | 0.8452 |
| 38 | 4.0 | 0.8368 |

**Across all 33 pairs:**
- Mean F1: 0.844 (high similarity)
- Range: 0.821 - 0.870
- Subquestion-level variance within questions was small (all 3 subquestions for a given question scored within ~0.02 of each other)

### ROUGE-L F1

| Question ID | Psychologist Mean | ROUGE-L F1 |
|-------------|-----------------|------------|
| 1 | 4.2 | 0.1593 |
| 3 | 4.3 | 0.1510 |
| 5 | 4.0 | 0.1894 |
| 11 | 4.1 | 0.1554 |
| 12 | 4.2 | 0.1628 |
| 13 | 4.2 | 0.1733 |
| 18 | 4.0 | 0.1390 |
| 19 | 4.0 | 0.1287 |
| 25 | 4.0 | 0.2048 |
| 34 | 4.2 | 0.1561 |
| 38 | 4.0 | 0.1550 |

**Across all 33 pairs:**
- Mean F1: 0.162 (low overlap)
- Range: 0.118 - 0.239

## Correlation with Human Ratings

The key question: do BERTScore and ROUGE-L scores predict what psychologists rated as high-quality?

| Metric | Pearson r | p-value | Spearman rho | p-value |
|--------|-----------|---------|--------------|---------|
| BERTScore F1 | 0.135 | 0.691 | 0.152 | 0.656 |
| ROUGE-L F1 | -0.101 | 0.767 | 0.059 | 0.864 |

**None of the correlations are statistically significant** (all p-values >> 0.05). The automated metrics do not show meaningful correlation with psychologist ratings in this dataset.

## Analysis

### Why BERTScore is High but Uncorrelated

BERTScore F1 values are consistently high (0.82-0.87) across all questions, regardless of psychologist rating. This suggests the generated responses are semantically similar to the references in embedding space -- the model uses related concepts, related terms, and similar sentence structure.

However, ROUGE-L scores are low (0.12-0.20), meaning the generated responses share very few exact n-gram sequences with the references. The model paraphrases heavily rather than echoing the reference language directly.

The lack of correlation with human ratings could be explained by:

1. **Ceiling effect on BERTScore:** All questions in this analysis had psychologist ratings of 4.0-4.3 (already high-quality responses). The reference responses are all good, so the generated responses also score well on semantic similarity. There is not enough variance in psychologist ratings to establish a correlation.

2. **ROUGE-L measures surface overlap, not quality:** ROUGE-L is a surface-level metric. Two well-written responses on the same topic can have low ROUGE-L because they phrase things differently. Low ROUGE-L does not indicate low quality -- it indicates paraphrasing.

3. **Reference and candidate share the same topic and format:** Both reference and candidate responses are mythological mental health advice in the same style (bullet points, empathetic tone, story-first structure). This drives BERTScore up regardless of quality differences.

4. **Small sample size:** With only 11 questions in the correlation analysis, the study has limited statistical power to detect even moderate correlations.

## Key Findings

1. **BERTScore F1 is uniformly high** (mean 0.844) across all questions, indicating strong semantic alignment with references regardless of psychologist rating.

2. **ROUGE-L F1 is uniformly low** (mean 0.162), indicating significant paraphrasing -- the generated responses convey similar meaning but use different wording.

3. **No correlation with human ratings** -- neither metric shows statistically significant correlation with psychologist mean overall ratings in this dataset.

4. **The reference set is narrow** -- all 11 analyzed questions had psychologist ratings between 4.0 and 4.3. There are no low-rated questions in the comparison set to establish a quality gradient.

## Limitations

1. **Small sample size** (n=11) limits statistical power. With 19 questions in the study but only 11 meeting the >= 4.0 threshold, the correlation analysis is underpowered.

2. **All references are already high-quality** -- since references are drawn from questions with psychologist ratings >= 4.0, there is no coverage of the low-rated range. The correlation cannot capture how metrics behave across the full quality spectrum.

3. **BERTScore with roberta-large may be too permissive** -- for this particular task (mythological mental health advice), the model detects semantic similarity even when the specific story, teaching, or suggestion differs substantially. The metric captures topic similarity more than quality.

4. **ROUGE-L is not a quality metric** -- ROUGE-L measures surface-level n-gram overlap. For a paraphrasing task like this, low ROUGE-L is expected and does not indicate poor quality.

## Recommendations

1. **Widen the reference set** -- include questions with lower psychologist ratings (3.0-3.9) to establish whether BERTScore can distinguish between higher and lower quality responses.

2. **Consider alternative reference selection** -- instead of a hard threshold at 4.0, use the full range of psychologist ratings as continuous references, or use each response's mean rating as a weight in correlation analysis.

3. **Evaluate per-criterion correlation** -- instead of overall_perceived_helpfulness, correlate BERTScore/ROUGE-L with specific criteria like relevance_of_story or clarity_of_teaching, where the metrics may be more discriminative.

4. **Test with human-generated baselines** -- compute BERTScore/ROUGE-L for the baseline responses (no mythology) against the same references to establish whether the metrics can distinguish mythological from non-mythological responses.

## Technical Notes

- All generation was run with qwen3:8b via Ollama, temperature=0.0
- BERTScore model: roberta-large, layer 17, no IDF rescaling
- ROUGE-L: stemmed, using longest common subsequence
- Correlation computed at question level (3 subquestion scores averaged per question)
- 33 subquestion pairs, 11 questions, 5 psychologists per question