# Evaluation Protocol

## Design

Evaluate the same 40 primary questions for each of four models in two conditions: RAG and No-RAG. This produces 320 generated answers. Five psychologists independently score every answer, giving 200 ratings per model-condition-metric and 9,600 scalar ratings in total.

For each model, use an identical system prompt, user question, response length limit, sampling settings, and safety policy in both conditions. In the RAG condition, record the retrieval index/corpus version, embedding model, retrieval method, and top-k value. In the No-RAG condition, do not provide retrieved passages or citations. Record exact model identifiers, inference provider, temperature, seed when available, and generation date.

## Blinding and allocation

Create the answers before inviting reviewers. The packet builder assigns randomized `blind_answer_id` values and excludes model and condition labels from reviewer packets. Keep `rater_packets/private_assignment_key.csv` with the research team only. Do not share prior ratings, aggregate results, or the screenshot values with reviewers.

Each psychologist completes one full packet containing all 320 answers in randomized order. Reviewers may leave a brief optional comment, but should score every criterion for every answer. Use stable pseudonymous rater IDs such as `P01` to `P05`; do not put names in the research dataset.

## Rating rubric

Score each criterion from 1 (very poor) to 5 (excellent). Use whole numbers only.

| Criterion | What to assess |
|---|---|
| Empathy | Whether the response acknowledges the user's emotional experience with a respectful, supportive tone. |
| Relevance | Whether it directly addresses the specific question and context. |
| Clarity | Whether the explanation is understandable, organized, and free of avoidable ambiguity. |
| Actionability | Whether it offers practical, feasible next steps appropriate to the situation. |
| Justification | Whether suggestions are explained with sound reasoning rather than asserted without support. |
| Helpfulness | Overall expected usefulness to the user, considering the response as a whole. |

Use `1` for seriously inadequate, `2` for weak, `3` for adequate but limited, `4` for strong, and `5` for excellent. Rate the answer actually shown, not the model or the RAG approach. Do not infer quality from writing style alone.

## Safety handling

If an answer contains an urgent safety concern, flag it in the comment field and follow your institution's review procedure. The rating should still reflect the answer shown. Keep raw responses and reviewer data access-controlled because the prompts concern mental-health experiences.

## Analysis

For each model-condition pair and each criterion, calculate the arithmetic mean across all 40 questions and all five psychologists. Do not replace missing scores with zero. The final Table 5 is complete only when every cell is based on 200 valid ratings. Retain the raw packets, private assignment key, response manifest, and script output so the analysis is reproducible.
