import pandas as pd
import ollama
import time
import os
from tqdm import tqdm

def generate_responses(input_csv, output_csv, model_name='qwen3:8b'):
    print(f"Loading questions from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    results = []
    
    prompt_template = """You are a mental health counselor chatbot drawing from Indian mythological stories 
(Ramayana, Mahabharata) to provide culturally-grounded emotional support.

Question: {question_text}

Please provide a thoughtful, empathetic response that:
1. Validates the user's feelings
2. Weaves in a relevant mythological story (Ramayana or Mahabharata)
3. Draws a clear parallel between the story and the user's situation
4. Ends with 2-3 actionable suggestions

Keep the tone warm, supportive, and clear. Response should be 150-300 words."""

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print(f"Generating responses using {model_name}...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        q_text = row['question_text']
        prompt = prompt_template.format(question_text=q_text)
        
        success = False
        retries = 3
        delay = 0.5
        
        while not success and retries > 0:
            try:
                start_time = time.time()
                response = ollama.chat(model=model_name, messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ], options={'temperature': 0.0})
                
                generation_time = time.time() - start_time
                generated_text = response['message']['content']
                tokens_used = response.get('eval_count', 0)
                
                results.append({
                    'question_id': row['question_id'],
                    'subquestion_id': row['subquestion_id'],
                    'prompt': prompt,
                    'generated_response': generated_text,
                    'tokens_used': tokens_used,
                    'generation_time_seconds': generation_time
                })
                success = True
                time.sleep(delay)  # Rate limiting delay
            except Exception as e:
                retries -= 1
                print(f"Error generating response for Question {row['question_id']}, Subquestion {row['subquestion_id']}: {e}")
                if retries > 0:
                    print(f"Retrying... ({retries} attempts left)")
                    time.sleep(2 ** (3 - retries))  # Exponential backoff
                else:
                    print("Max retries reached. Skipping.")
                    results.append({
                        'question_id': row['question_id'],
                        'subquestion_id': row['subquestion_id'],
                        'prompt': prompt,
                        'generated_response': f"ERROR: Failed to generate response. Details: {e}",
                        'tokens_used': 0,
                        'generation_time_seconds': 0.0
                    })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved generated responses to {output_csv}")
    print(f"Total processed: {len(results_df)}")
    
    errors = results_df[results_df['generated_response'].str.startswith('ERROR:')]
    if len(errors) > 0:
        print(f"Warning: {len(errors)} subquestions failed to generate.")

if __name__ == "__main__":
    generate_responses(
        input_csv='test_questions.csv',
        output_csv='data/generated_new_responses_qwen3.csv',
        model_name='qwen3:8b'
    )
