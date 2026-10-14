#!/usr/bin/env python3
"""Run Qwen3-8B through the SAME benchmark (identical RAG / No-RAG prompts, retrieval and settings).

Only difference: Qwen3's hidden reasoning ("thinking") is switched off and any <think> block is removed,
so the saved answer is the user-facing response, as for the other four models.
If the GPU runner crashes (CUDA error), the call is retried, then falls back to CPU for that answer.
"""
import re, sys, time
import ollama
import run_benchmark as rb

THINK = re.compile(r"<think>.*?</think>", re.S)
_cpu_only = False


def _chat(model_tag, prompt, opts):
    try:
        return ollama.chat(model=model_tag, messages=[{"role": "user", "content": prompt}], options=opts, think=False)
    except TypeError:  # older ollama client without the think parameter
        return ollama.chat(model=model_tag, messages=[{"role": "user", "content": prompt + "\n/no_think"}], options=opts)


def generate(model_tag, prompt, temperature):
    global _cpu_only
    started = time.perf_counter()
    base = {"temperature": temperature}
    attempts = ([] if _cpu_only else [dict(base), dict(base, num_ctx=4096)]) + [dict(base, num_gpu=0)]
    last = None
    for opts in attempts:
        try:
            result = _chat(model_tag, prompt, opts)
            if opts.get("num_gpu") == 0 and not _cpu_only:
                _cpu_only = True
                print("  [note] GPU runner keeps failing - continuing on CPU (slower, same model).", flush=True)
            text = THINK.sub("", result["message"]["content"]).strip()
            return text, int(result.get("eval_count", 0)), time.perf_counter() - started
        except Exception as exc:
            last = exc
            print(f"  [retry] {str(exc)[:110]}", flush=True)
            time.sleep(8)
    raise last


rb.generate = generate
if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--config", str(rb.ROOT / "benchmark_config_qwen3.json")] + sys.argv[1:]
    rb.main()
