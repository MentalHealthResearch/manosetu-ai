import pandas as pd
import os
from pathlib import Path

def _resolve_path(filename: str) -> str:
    """Resolve a filename relative to this script's parent directory (project root)."""
    script_dir = Path(__file__).resolve().parent.parent  # src/ -> project root
    return str(script_dir / filename)

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

def build_references(evaluations_csv, answers_csv, output_csv):
    # Resolve paths relative to project root
    evaluations_csv = _resolve_path(evaluations_csv)
    answers_csv = _resolve_path(answers_csv)
    output_csv = _resolve_path(output_csv)

    print(f"Loading psychologist evaluations from {evaluations_csv}...")
    eval_df = pd.read_csv(evaluations_csv)
    
    # Calculate the mean of the 5 psychologists for overall_perceived_helpfulness
    overall_cols = _find_overall_helpfulness_columns(eval_df)
    print(f"Using columns: {overall_cols}")
    
    eval_df['psychologist_mean_overall'] = eval_df[overall_cols].mean(axis=1)
    eval_df['is_high_quality'] = eval_df['psychologist_mean_overall'] >= 3.5
    
    # Filter for high quality
    high_quality_qs = eval_df[eval_df['is_high_quality']][['question_id', 'psychologist_mean_overall', 'is_high_quality']]
    print(f"Found {len(high_quality_qs)} high-quality questions (mean rating >= 3.5).")
    
    print(f"Loading answers from {answers_csv}...")
    ans_df = pd.read_csv(answers_csv)
    
    # Use subquestion_id = 1 as the reference text for the whole question
    primary_ans_df = ans_df[ans_df['subquestion_id'] == 1].copy()
    primary_ans_df = primary_ans_df.rename(columns={'response_c': 'reference_text'})
    
    # Join to get the reference text
    merged_df = pd.merge(high_quality_qs, primary_ans_df[['question_id', 'reference_text']], on='question_id', how='inner')
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged_df.to_csv(output_csv, index=False)
    print(f"Saved {len(merged_df)} psychologist references to {output_csv}")

if __name__ == "__main__":
    build_references(
        evaluations_csv='Final_human_evaluations_improved.csv',
        answers_csv='archive/test_questions_answers.csv',
        output_csv='data/psychologist_references.csv'
    )
