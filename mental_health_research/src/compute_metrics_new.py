import pandas as pd
import os
from bert_score import score
from rouge_score.rouge_scorer import RougeScorer
from tqdm import tqdm

def compute_metrics(references_csv, candidates_csv, output_dir):
    print(f"Loading references from {references_csv}...")
    refs_df = pd.read_csv(references_csv)
    
    print(f"Loading candidates from {candidates_csv}...")
    cand_df = pd.read_csv(candidates_csv)
    
    # Join on question_id
    # Note: Candidates have subquestion_id, but references only have question_id.
    # The join on question_id will match the same reference text to all subquestions of that question.
    merged = pd.merge(cand_df, refs_df, on='question_id', how='inner')
    
    if len(merged) == 0:
        print("Warning: No overlapping rows found. Check your input files.")
        return
        
    print(f"Found {len(merged)} candidate-reference pairs to evaluate.")
    
    candidates = merged['generated_response'].astype(str).tolist()
    references = merged['reference_text'].astype(str).tolist()
    
    bert_results = []
    rouge_results = []
    
    # ROUGE
    print("Computing ROUGE-L metrics...")
    rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)
    for i, (cand, ref) in tqdm(enumerate(zip(candidates, references)), total=len(candidates)):
        rouge_scores = rouge_scorer.score(ref, cand)
        rouge_results.append({
            'question_id': merged['question_id'].iloc[i],
            'subquestion_id': merged.get('subquestion_id', pd.Series([None]*len(merged))).iloc[i],
            'rougeL_precision': rouge_scores['rougeL'].precision,
            'rougeL_recall': rouge_scores['rougeL'].recall,
            'rougeL_f1': rouge_scores['rougeL'].fmeasure
        })
    
    # BERTScore
    print("Computing BERTScore (roberta-large, layer 17)...")
    P, R, F1 = score(candidates, references, model_type="roberta-large", num_layers=17, verbose=True)
    
    for i in range(len(candidates)):
        bert_results.append({
            'question_id': merged['question_id'].iloc[i],
            'subquestion_id': merged.get('subquestion_id', pd.Series([None]*len(merged))).iloc[i],
            'bert_precision': P[i].item(),
            'bert_recall': R[i].item(),
            'bert_f1': F1[i].item()
        })
        
    os.makedirs(output_dir, exist_ok=True)
    
    bert_df = pd.DataFrame(bert_results)
    bert_out = os.path.join(output_dir, 'bertscore_new_responses.csv')
    bert_df.to_csv(bert_out, index=False)
    print(f"Saved BERTScore results to {bert_out}")
    print(f"Mean BERTScore F1: {bert_df['bert_f1'].mean():.4f}")
    
    rouge_df = pd.DataFrame(rouge_results)
    rouge_out = os.path.join(output_dir, 'rouge_new_responses.csv')
    rouge_df.to_csv(rouge_out, index=False)
    print(f"Saved ROUGE results to {rouge_out}")
    print(f"Mean ROUGE-L F1: {rouge_df['rougeL_f1'].mean():.4f}")

if __name__ == "__main__":
    compute_metrics(
        references_csv='data/psychologist_references.csv',
        candidates_csv='data/generated_new_responses_qwen3.csv',
        output_dir='data/'
    )
