"""Stage 2-4 - dense (baseline or bge-m3), HyDE dense, BM25, RRF fusion, construct boost, cross-encoder rerank."""
from __future__ import annotations
import numpy as np
from bm25 import BM25
from constructs import SYNONYMS
from corpus import retrieval_text, rerank_text

# Ablation ladder. Each config adds one component to the previous one.
CONFIGS = {
    "A0_baseline_mpnet":    dict(encoder="baseline", hyde=False, bm25=False, boost=False, rerank=False),
    "A1_bge_m3":            dict(encoder="main", hyde=False, bm25=False, boost=False, rerank=False),
    "A2_+query_enrichment": dict(encoder="main", hyde=True,  bm25=False, boost=False, rerank=False),
    "A3_+hybrid_bm25":      dict(encoder="main", hyde=True,  bm25=True,  boost=False, rerank=False),
    "A4_+construct_boost":  dict(encoder="main", hyde=True,  bm25=True,  boost=True,  rerank=False),
    "A5_+rerank_full":      dict(encoder="main", hyde=True,  bm25=True,  boost=True,  rerank=True),
}


class HybridRetriever:
    def __init__(self, records, encoders: dict, reranker, cfg: dict):
        self.records, self.enc, self.reranker, self.cfg = records, encoders, reranker, cfg
        texts = [retrieval_text(r) for r in records]
        self.doc_vecs = {k: self._cached(e, texts) for k, e in encoders.items()}
        self.bm25 = BM25(texts)
        self.rr_texts = [rerank_text(r) for r in records]
        self.cons = [set(r["constructs"]) for r in records]

    @staticmethod
    def _cached(enc, texts):
        """Encode the corpus once per model and reuse it in later runs (outputs/emb_cache)."""
        import hashlib
        from common import OUT
        key = hashlib.md5(("\n".join(texts) + enc.name + str(enc.mock)).encode("utf-8")).hexdigest()[:12]
        d = OUT / "emb_cache"; d.mkdir(exist_ok=True)
        f = d / f"{enc.name.replace('/', '_')}_{key}.npy"
        if f.exists():
            print(f"[cache] corpus embeddings loaded: {f.name}", flush=True)
            return np.load(f)
        v = enc.encode(texts); np.save(f, v)
        return v

    @staticmethod
    def _ranks(scores: np.ndarray) -> np.ndarray:
        order = np.argsort(-scores)
        ranks = np.empty_like(order); ranks[order] = np.arange(1, len(order) + 1)
        return ranks

    def search(self, query: str, qu: dict | None, conf: dict) -> list[dict]:
        k_rrf, pool = self.cfg["rrf_k"], self.cfg["candidate_pool"]
        vecs = self.doc_vecs[conf["encoder"]]
        enc = self.enc[conf["encoder"]]
        lists = [vecs @ enc.encode([query])[0]]
        dense_raw = lists[0]
        if conf["hyde"] and qu and qu.get("hypothetical_episode"):
            lists.append(vecs @ enc.encode([qu["hypothetical_episode"] + " " + qu.get("situation", "")])[0])
        if conf["bm25"]:
            expansion = " ".join(s for c in (qu or {}).get("constructs", []) for s in SYNONYMS.get(c, []))
            lists.append(np.asarray(self.bm25.scores(f"{query} {(qu or {}).get('situation', '')} {expansion}")))
        if len(lists) == 1:
            fused = dense_raw.copy()
        else:
            fused = sum(1.0 / (k_rrf + self._ranks(s)) for s in lists)
        if conf["boost"] and qu:
            want = set(qu.get("constructs", []))
            fused = fused + self.cfg["construct_boost"] * np.array([len(want & c) for c in self.cons])
        cand = list(np.argsort(-fused)[:pool])
        out = [{"idx": int(i), "story_id": self.records[i]["story_id"], "fused": float(fused[i]),
                "dense": float(dense_raw[i])} for i in cand]
        if conf["rerank"]:
            q = query + (f" Situation: {qu['situation']}" if qu and qu.get("situation") else "")
            scores = self.reranker.score(q, [self.rr_texts[c["idx"]] for c in out])
            for c, s in zip(out, scores):
                c["rerank"] = float(s)
            out.sort(key=lambda c: -c["rerank"])
        return out
