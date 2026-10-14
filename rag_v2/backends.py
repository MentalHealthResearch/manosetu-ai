"""Model back-ends: sentence-transformer encoders, cross-encoder reranker, Ollama LLM.

`--mock` swaps every model for a tiny deterministic stand-in so the whole pipeline can be
smoke-tested without GPUs or downloads. Mock results are meaningless as numbers.
"""
from __future__ import annotations
import hashlib, json, os, re
import numpy as np
from bm25 import tokenize
from constructs import CONSTRUCTS, SYNONYMS


class Encoder:
    def __init__(self, name: str, mock: bool = False):
        self.name, self.mock = name, mock
        if not mock:
            import torch
            from sentence_transformers import SentenceTransformer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[encoder] {name} on {self.device.upper()}"
                  + ("" if self.device == "cuda" else "  (no CUDA in this Python env - this step will be slow)"), flush=True)
            self.model = SentenceTransformer(name, device=self.device)
            self.model.max_seq_length = min(512, self.model.max_seq_length or 512)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.mock:
            vecs = np.zeros((len(texts), 512), dtype=np.float32)
            for i, t in enumerate(texts):
                for tok in tokenize(t):
                    vecs[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 512] += 1
            n = np.linalg.norm(vecs, axis=1, keepdims=True); n[n == 0] = 1
            return vecs / n
        return np.asarray(self.model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=len(texts) > 50))


class Reranker:
    def __init__(self, name: str, mock: bool = False):
        self.mock = mock
        if not mock:
            import torch
            from sentence_transformers import CrossEncoder
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[reranker] {name} on {dev.upper()}", flush=True)
            self.model = CrossEncoder(name, max_length=512, device=dev)
        else:
            self.enc = Encoder("mock", True)

    def score(self, query: str, docs: list[str]) -> list[float]:
        if self.mock:
            q = self.enc.encode([query])[0]; d = self.enc.encode(docs)
            return (d @ q).tolist()
        s = self.model.predict([(query, d) for d in docs], show_progress_bar=False)
        s = np.asarray(s, dtype=float)
        if s.min() < 0 or s.max() > 1:           # logits -> probability
            s = 1 / (1 + np.exp(-s))
        return s.tolist()


class LLM:
    def __init__(self, tag: str, temperature: float = 0.0, mock: bool = False):
        self.tag, self.temperature, self.mock = tag, temperature, mock

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        if self.mock:
            return self._mock(messages, json_mode)
        import ollama
        # JSON mode at temperature 0 can loop on repeated tokens; cap length, penalise repeats and
        # retry with a little sampling, finally without JSON mode (parse_json extracts the object).
        attempts = [(json_mode, self.temperature, 1.1), (json_mode, 0.3, 1.2), (False, 0.3, 1.2)]
        last = None
        for use_json, temp, rep in attempts:
            opts = {"temperature": temp, "repeat_penalty": rep, "num_predict": 450 if json_mode else 700, "seed": 42}
            kw = {"format": "json"} if use_json else {}
            try:
                r = ollama.chat(model=self.tag, messages=messages, options=opts, **kw)
                return r["message"]["content"]
            except Exception as exc:          # ollama.ResponseError: token repeat limit etc.
                last = exc
                print(f"  [retry] {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        if json_mode:
            return "{}"                        # caller falls back to safe defaults
        raise last

    def _mock(self, messages, json_mode):
        text = messages[-1]["content"].lower()
        if json_mode and "unsupported_claims" in text:
            return json.dumps({"faithful": True, "unsupported_claims": []})
        if json_mode:
            q = text.split("user message:")[-1]
            hits = [c for c in CONSTRUCTS if any(s in q for s in SYNONYMS[c])] or ["Stress Management"]
            return json.dumps({"constructs": hits[:2], "risk": "none", "situation": q[:120],
                               "hypothetical_episode": "A hero feels " + " ".join(SYNONYMS[hits[0]][:4])})
        return "Mock response. It is understandable to feel this way. Try one small step today."


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        return json.loads(m.group(0)) if m else {}
