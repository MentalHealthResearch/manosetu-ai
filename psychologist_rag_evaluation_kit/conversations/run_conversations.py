#!/usr/bin/env python3
"""Multi-turn RAG vs No-RAG conversations (one per construct) with a single model.

User turns are scripted (scripted_user_turns.json) and identical in both conditions; every
assistant turn is generated live by the model. RAG retrieves top-3 passages per turn from the
same Qdrant collection used in the benchmark. Output: conversations_<model>.json (resumable).
"""
import argparse, json, os, time
from datetime import datetime, timezone
from pathlib import Path
import ollama

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE.parent / "benchmark" / "benchmark_config.json").read_text(encoding="utf-8"))

def as_text(v):
    return "; ".join(map(str, v)) if isinstance(v, list) else str(v or "")

SYSTEM_BASE = ("You are a supportive mental-health conversation assistant. Provide culturally sensitive, "
               "non-judgmental support. Validate the person's feelings, respond to what they said earlier in the "
               "conversation, and give 1-3 practical next steps. Do not diagnose. If there is immediate danger or "
               "self-harm risk, gently check on safety and encourage local emergency or crisis support. "
               "Reply conversationally in at most 120 words.")
SYSTEM_RAG = SYSTEM_BASE + (" Use only the supplied narrative passages when making a narrative reference; do not invent "
               "story details and do not mention 'passages' or sources by number. Make a specific connection if relevant.")
SYSTEM_NORAG = SYSTEM_BASE + " Do not invent narrative references and do not claim access to a knowledge base."

def passages_block(ps):
    out = []
    for i, p in enumerate(ps, 1):
        out.append(f"[Passage {i}; source ID: {as_text(p.get('story_id'))}]\nSource: {as_text(p.get('corpus'))}\n"
                   f"Text: {as_text(p.get('paragraph'))}\nPsychological summary: {as_text(p.get('psych_summary'))}\n"
                   f"Key teaching/action: {as_text(p.get('key_teaching'))}")
    return "\n\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-label", default="Gemma-2-9B")
    args = ap.parse_args()
    tag = CFG["models"][args.model_label]
    out_path = HERE / f"conversations_{args.model_label}.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    have = {(d["construct"], d["condition"]) for d in done}
    scripts = json.loads((HERE / "scripted_user_turns.json").read_text(encoding="utf-8"))

    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    os.environ.setdefault("HF_HOME", str(HERE.parent.parent / "mental_health_research" / ".hf_cache"))
    client = QdrantClient(url=CFG["qdrant_url"])
    enc = SentenceTransformer(CFG["embedding_model"])

    def retrieve(q):
        v = enc.encode(q, normalize_embeddings=True).tolist()
        res = client.query_points(collection_name=CFG["qdrant_collection"], query=v, limit=int(CFG["top_k"]), with_payload=True)
        return [{**(p.payload or {}), "retrieval_score": p.score} for p in res.points]

    for s in scripts:
        for cond in ["RAG", "No-RAG"]:
            if (s["construct"], cond) in have:
                print("skip", s["construct"], cond); continue
            history, turns = [], []
            for ti, user in enumerate(s["turns"], 1):
                if cond == "RAG":
                    query = user if ti == 1 else s["turns"][0] + " " + user
                    ps = retrieve(query)
                    system = SYSTEM_RAG + "\n\nRetrieved narrative passages:\n" + passages_block(ps)
                else:
                    ps, system = [], SYSTEM_NORAG
                msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user}]
                t0 = time.perf_counter()
                r = ollama.chat(model=tag, messages=msgs, options={"temperature": float(CFG["temperature"])})
                reply = r["message"]["content"].strip()
                history += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
                turns.append({"turn": ti, "user": user, "assistant": reply,
                              "retrieved_story_ids": [as_text(p.get("story_id")) for p in ps],
                              "retrieval_scores": [round(p["retrieval_score"], 4) for p in ps],
                              "seconds": round(time.perf_counter() - t0, 2)})
                print(f"{s['construct']} | {cond} | turn {ti} done", flush=True)
            done.append({"construct": s["construct"], "qid": s["qid"], "condition": cond, "model": args.model_label,
                         "model_tag": tag, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "turns": turns})
            out_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {len(done)} conversations to {out_path}")

if __name__ == "__main__":
    main()
