import os
import subprocess
import sys

def run_script(script_path):
    print("=" * 60)
    print(f"Running {script_path}...")
    print("=" * 60)
    
    # Use the current python executable
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        print(f"Error: {script_path} failed with return code {result.returncode}")
        sys.exit(1)
        
    print(f"Successfully finished {script_path}\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, 'src')
    os.chdir(base_dir)
    
    scripts = [
        'generate_responses_rag.py',
        'build_references_from_psychologists.py',
        'compute_metrics_rag.py',
        'correlation_analysis_rag.py',
    ]
    
    for script in scripts:
        script_path = os.path.join(src_dir, script)
        if not os.path.exists(script_path):
            print(f"Warning: {script_path} not found, skipping.")
            continue
        run_script(script_path)
        
    print("Full pipeline completed successfully!")
