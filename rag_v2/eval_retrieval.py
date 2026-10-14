#!/usr/bin/env python3
"""Retrieval ablation A0-A5 on the 40 benchmark questions.

Metrics on the 30 construct-labelled ground-truth queries (silver, construct level):
  C-P@1, C-P@3  : share of top-1 / top-3 episodes annotated with the query's gold construct
  C-MRR@10      : reciprocal rank of the first construct-matching episode
  ConstructAcc  : whether the LLM detected the gold construct (query-understanding quality)
Episode-level Recall@3 / MRR are added automatically once expected_episode_id is filled in.
Metrics on all 40 queries: distinct top-3 episodes (diversity), mean top-1 dense similarity.
"""
from __future__ import annotations
import argparse, json
import pandas as pd
from common import CFG, OUT, path, load_qu_cache, save_qu_cache
from corpus import load_corpus
from constructs import normalise
from backends import Encoder, Reranker, LLM
from query_understanding import understand
from retriever import HybridRetriever, CONFIGS


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock", action="store_true"); a = ap.parse_args()
    recs = load_corpus(path("dataset_root"))
    qs = pd.read_csv(path("questions_csv"))
    gt = pd.read_csv(path("ground_truth_csv"))
    gold = {int(r.question_id): normalise(r.construct) for r in gt.itertuples()}
    exp = {int(r.question_id): {s.strip() for s in str(r.expected_episode_id).split(";") if s.strip() and s.strip() != "nan"}
           for r in gt.itertuples()}
    print(f"Corpus: {len(recs)} episodes; questions: {len(qs)}; labelled: {len(gold)}")

    llm = LLM(CFG["llm_model"], CFG["temperature"], a.mock)
    cache = load_qu_cache(a.mock)
    for r in qs.itertuples():
        k = str(r.question_id)
        if k not in cache:
            cache[k] = understand(llm, r.question_text); save_qu_cache(cache, a.mock)
            print("QU", k, cache[k]["constructs"], cache[k]["risk"])

    encs = {"baseline": Encoder(CFG["baseline_embedding_model"], a.mock), "main": Encoder(CFG["embedding_model"], a.mock)}
    rr = Reranker(CFG["reranker_model"], a.mock)
    ret = HybridRetriever(recs, encs, rr, CFG)

    rows, summary = [], []
    for name, conf in CONFIGS.items():
        per = []
        for r in qs.itertuples():
            qid = int(r.question_id); qu = cache[str(qid)]
            res = ret.search(r.question_text, qu if (conf["hyde"] or conf["bm25"] or conf["boost"]) else None, conf)
            top = res[:CFG["final_k"]]
            rec = {"config": name, "question_id": qid, "top_ids": ";".join(c["story_id"] for c in top),
                   "top1_dense": top[0]["dense"], "top1_rerank": top[0].get("rerank")}
            if qid in gold:
                g = gold[qid]
                hits = [bool(g & set(recs[c["idx"]]["constructs"])) for c in res[:10]]
                rec |= {"C_P@1": float(hits[0]), "C_P@3": sum(hits[:3]) / 3,
                        "C_MRR@10": next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0),
                        "construct_detected": float(bool(g & set(qu["constructs"])))}
                if exp[qid]:
                    ids = [c["story_id"] for c in res]
                    rec |= {"Recall@3": float(bool(exp[qid] & set(ids[:3]))),
                            "MRR@20": next((1 / (i + 1) for i, s in enumerate(ids) if s in exp[qid]), 0.0)}
            per.append(rec)
        df = pd.DataFrame(per); rows.append(df)
        lab = df.dropna(subset=["C_P@1"])
        s = {"config": name, "C_P@1": lab["C_P@1"].mean(), "C_P@3": lab["C_P@3"].mean(), "C_MRR@10": lab["C_MRR@10"].mean(),
             "distinct_top3_episodes_40q": len({i for x in df.top_ids for i in x.split(";")}),
             "mean_top1_dense_sim": df.top1_dense.mean()}
        if "Recall@3" in df:
            s |= {"Recall@3": df["Recall@3"].mean(), "MRR@20": df["MRR@20"].mean()}
        summary.append(s); print(s)
    tag = "_mock" if a.mock else ""
    pd.concat(rows).to_csv(OUT / f"retrieval_per_query{tag}.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / f"retrieval_ablation_summary{tag}.csv", index=False)
    acc = pd.concat(rows).query("config=='A5_+rerank_full'")["construct_detected"].mean()
    (OUT / f"retrieval_ablation_summary{tag}.md").write_text(
        md_table(sm.round(3)) + f"\n\nConstruct detection accuracy (LLM vs gold, 30 queries): {acc:.3f}\n"
        "Note: C-metrics use the construct annotation of each episode as a silver relevance label. A4/A5 use the detected "
        "construct as a boost, so their C-metrics partly reflect detection accuracy; episode-level Recall@3/MRR from the "
        "expert ground-truth file is the unbiased test.\n", encoding="utf-8")
    print("Saved", OUT)


if __name__ == "__main__":
    main()
