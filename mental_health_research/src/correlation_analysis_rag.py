#!/usr/bin/env python3
"""
correlation_analysis_rag.py — Correlation analysis for RAG-backed responses.

Adapted from correlation_analysis_new.py.
Reads data/bertscore_rag.csv, data/rouge_rag.csv, Final_human_evaluations_improved.csv.
Outputs data/metrics_summary_rag.csv and data/correlation_report_rag.md.
"""

import argparse
import os

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


def _find_overall_helpfulness_columns(eval_df: pd.DataFrame) -> list[str]:
    """Find all psychologist overall helpfulness columns.
    
    Raises ValueError with a clear message if neither spelling exists.
    """
    candidates = [
        c for c in eval_df.columns
        if "overall_perceived_helpfulness" in c or "overall_perceoved_helpfulness" in c
    ]
    if not candidates:
        available = [c for c in eval_df.columns if "overall" in c.lower()]
        raise ValueError(
            f"No 'overall_perceived_helpfulness' columns found. "
            f"Closest matches: {available}"
        )
    return candidates


def run_correlation(bert_csv: str, rouge_csv: str, human_csv: str, output_dir: str):
    print("Loading metric results...")
    bert_df = pd.read_csv(bert_csv)
    rouge_df = pd.read_csv(rouge_csv)

    print("Loading human evaluations...")
    human_df = pd.read_csv(human_csv)

    # Calculate the mean of the 5 psychologists for overall_perceived_helpfulness
    overall_cols = _find_overall_helpfulness_columns(human_df)
    print(f"Using columns: {overall_cols}")

    human_df["psychologist_mean_overall"] = human_df[overall_cols].mean(axis=1)
    human_ratings = human_df[["question_id", "psychologist_mean_overall"]]

    # Aggregate subquestion metrics to question level
    bert_agg = bert_df.groupby("question_id")["bert_f1"].mean().reset_index()
    rouge_agg = rouge_df.groupby("question_id")["rougeL_f1"].mean().reset_index()

    # Merge all
    merged = pd.merge(human_ratings, bert_agg, on="question_id", how="inner")
    merged = pd.merge(merged, rouge_agg, on="question_id", how="inner")

    if len(merged) == 0:
        print("No matching data found for correlation analysis.")
        return

    # Save per-question metrics
    os.makedirs(output_dir, exist_ok=True)
    summary_out = os.path.join(output_dir, "metrics_summary_rag.csv")
    merged.to_csv(summary_out, index=False)
    print(f"Saved metrics summary to {summary_out}")

    # Calculate correlations
    bert_f1 = merged["bert_f1"]
    rouge_f1 = merged["rougeL_f1"]
    human_score = merged["psychologist_mean_overall"]

    bert_pearson, bert_p_p = pearsonr(bert_f1, human_score)
    bert_spearman, bert_s_p = spearmanr(bert_f1, human_score)

    rouge_pearson, rouge_p_p = pearsonr(rouge_f1, human_score)
    rouge_spearman, rouge_s_p = spearmanr(rouge_f1, human_score)

    # Generate report
    report = f"""# Correlation Analysis Report (RAG-Backed)

## Overview
This report compares automated metrics (BERTScore and ROUGE-L) for **RAG-backed** responses
against human psychologist ratings.

Number of questions analyzed: {len(merged)}

## BERTScore vs. Human Ratings
- **Pearson Correlation:** {bert_pearson:.4f} (p-value: {bert_p_p:.4e})
- **Spearman Correlation:** {bert_spearman:.4f} (p-value: {bert_s_p:.4e})

## ROUGE-L vs. Human Ratings
- **Pearson Correlation:** {rouge_pearson:.4f} (p-value: {rouge_p_p:.4e})
- **Spearman Correlation:** {rouge_spearman:.4f} (p-value: {rouge_s_p:.4e})

## Interpretation
- A positive correlation indicates that the automated metric tends to increase when human ratings increase.
- Spearman correlation measures rank-order correlation (monotonicity), which is often more appropriate for ratings than Pearson (linear).
- These metrics now reflect **RAG-grounded** generation quality, not just parametric model knowledge.
"""

    report_out = os.path.join(output_dir, "correlation_report_rag.md")
    with open(report_out, "w") as f:
        f.write(report)

    print(f"Saved correlation report to {report_out}")
    print("\n--- Summary ---")
    print(f"BERTScore Spearman: {bert_spearman:.4f}")
    print(f"ROUGE-L Spearman:   {rouge_spearman:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Correlation analysis for RAG responses")
    parser.add_argument("--bert", default="data/bertscore_rag.csv", help="BERTScore CSV")
    parser.add_argument("--rouge", default="data/rouge_rag.csv", help="ROUGE CSV")
    parser.add_argument("--human", default="Final_human_evaluations_improved.csv", help="Human evaluations CSV")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()

    run_correlation(args.bert, args.rouge, args.human, args.output_dir)


if __name__ == "__main__":
    main()
