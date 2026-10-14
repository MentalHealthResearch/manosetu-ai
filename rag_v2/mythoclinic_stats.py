"""mythoclinic_stats.py - statistical procedures of Appendix H, run on the released result files.

Unit of analysis = question. Persona scores are averaged per response; each test compares two
responses to the same question. Wilcoxon signed-rank, two-sided, zero differences discarded.
Holm correction over the 5 models x 6 criteria RAG vs No-RAG family. The RAG1-vs-RAG2 comparison
(Gemma-2-9B) is exploratory and uncorrected. Effect size r_rb = (R+ - R-)/(R+ + R-).
Bootstrap: 10,000 percentile resamples of questions, seed 20261012. Top-1 retrieval: exact McNemar.
Anchor rows (responses re-scored in later runs to check drift) are excluded.

Usage (from the repository root):
    python rag_v2/mythoclinic_stats.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata, binomtest

ROOT = Path(__file__).resolve().parents[1] / "results"
CRIT = ["empathy", "relevance", "clarity", "actionability", "justification", "helpfulness"]
MODELS = ["Llama-3.1-8B", "Mistral-7B", "Gemma-2-9B", "MentalLLaMA-13B", "Qwen3-8B"]
SEED, B = 20261012, 10_000


def wsr(diff):
    """Two-sided Wilcoxon signed-rank, zeros discarded; returns p and rank-biserial r."""
    d = np.asarray(diff, float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0, 0.0
    r = rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return wilcoxon(d, zero_method="wilcox").pvalue, (rp - rm) / (rp + rm)


def boot_ci(diff):
    d = np.asarray(diff, float)
    rng = np.random.default_rng(SEED)
    means = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def holm(p):
    p = np.asarray(p)
    adj, run = np.empty_like(p), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def per_question(df):
    m = df.groupby(["model", "condition", "question_id"])[CRIT].mean()
    m["overall"] = m[CRIT].mean(axis=1)
    return m


def load():
    j = pd.read_csv(ROOT / "judge_ratings.csv")
    c = pd.read_csv(ROOT / "cas_scores.csv")
    return j[j.anchor == 0], c[c.anchor == 0]


def rag_vs_norag(j):
    m = per_question(j[j.condition.isin(["RAG", "No-RAG"])])
    rows = []
    for mod in MODELS:
        a = m.loc[(mod, "RAG")]
        b = m.loc[(mod, "No-RAG")].loc[a.index]
        for c in CRIT:
            p, r = wsr(a[c] - b[c])
            rows.append(dict(model=mod, criterion=c, rag=a[c].mean(), norag=b[c].mean(),
                             diff=(a[c] - b[c]).mean(), p=p, r_rb=r))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.values)
    return out


def rag1_vs_rag2(j, cas):
    rows = []
    m = per_question(j[j.model == "Gemma-2-9B"])
    a = m.loc[("Gemma-2-9B", "RAG-v2")]
    b = m.loc[("Gemma-2-9B", "RAG")].loc[a.index]
    for name, c in [("Overall judge score", "overall"), ("Helpfulness", "helpfulness"), ("Justification", "justification")]:
        p, r = wsr(a[c] - b[c]); lo, hi = boot_ci(a[c] - b[c])
        rows.append((name, b[c].mean(), a[c].mean(), p, r, lo, hi))
    cs = cas[cas.model == "Gemma-2-9B"].set_index(["condition", "question_id"])
    for name, c in [("CAS (all answers)", "CAS"), ("CAS situational relevance", "cas2_relevance")]:
        x = cs.loc["RAG-v2", c]
        y = cs.loc["RAG", c].loc[x.index]
        p, r = wsr(x - y); lo, hi = boot_ci(x - y)
        rows.append((name, y.mean(), x.mean(), p, r, lo, hi))
    g = pd.read_csv(ROOT / "grounding_audit_rag1_vs_rag2.csv").set_index(["system", "question_id"])
    both = g.loc["RAG1"].join(g.loc["RAG2"], lsuffix="_1", rsuffix="_2")
    both = both[(both.story_used_1 == 1) & (both.story_used_2 == 1)].sort_index()
    dq = both.parallel_quality_2 - both.parallel_quality_1
    p, r = wsr(dq); lo, hi = boot_ci(dq)
    rows.append((f"Parallel quality (n={len(dq)})", both.parallel_quality_1.mean(), both.parallel_quality_2.mean(), p, r, lo, hi))
    t = pd.read_csv(ROOT / "retrieval" / "retrieval_per_query.csv")
    t = t[t["C_P@1"].notna()]                                   # 30 construct-labelled questions
    a0 = t[t.config.str.startswith("A0")].set_index("question_id")["C_P@1"]
    a4 = t[t.config.str.startswith("A4")].set_index("question_id")["C_P@1"].loc[a0.index]
    b01, b10 = int(((a0 == 1) & (a4 == 0)).sum()), int(((a0 == 0) & (a4 == 1)).sum())
    pm = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    rows.append((f"Top-1 right construct (n={len(a0)})", a0.mean(), a4.mean(), pm, np.nan, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["metric", "RAG1", "RAG2", "p", "r_rb", "ci_lo", "ci_hi"])


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    j, cas = load()
    print("RAG vs No-RAG (Holm over 5 models x 6 criteria)")
    print(rag_vs_norag(j).round(3).to_string(index=False))
    print("\nRAG1 vs RAG2 (Gemma-2-9B; uncorrected)")
    print(rag1_vs_rag2(j, cas).round(3).to_string(index=False))
