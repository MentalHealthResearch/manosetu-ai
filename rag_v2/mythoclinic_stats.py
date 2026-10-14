"""mythoclinic_stats.py - statistical procedures of Appendix H.

Unit of analysis = question. Persona scores are averaged per response; each test compares two
responses to the same question. Wilcoxon signed-rank, two-sided, zero differences discarded.
Holm correction over the 4 models x 6 criteria RAG vs No-RAG family. RAG1-vs-RAG2 (Table 7) is
exploratory and uncorrected. Effect size r_rb = (R+ - R-)/(R+ + R-). Bootstrap: 10,000 percentile
resamples of questions, seed 20261012. Top-1 retrieval: exact McNemar test.

Usage: python mythoclinic_stats.py <ratings_4models.csv> <ratings_gemma_4cond.pkl>
       <cas_gemma_4cond.pkl> <grounding_both.pkl> <retrieval_per_query.csv>
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata, binomtest

CRIT = ["empathy", "relevance", "clarity", "actionability", "justification", "helpfulness"]
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
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def per_question(df, cond_col="condition"):
    m = df.groupby([cond_col, "question_id"])[CRIT].mean()
    m["overall"] = m[CRIT].mean(axis=1)
    return m


def rag_vs_norag(path):
    d = pd.read_csv(path)
    d = d[d.model.isin(["Llama-3.1-8B", "Mistral-7B", "Gemma-2-9B", "MentalLLaMA-13B"])]
    rows = []
    for mod, g in d.groupby("model"):
        m = per_question(g)
        a, b = m.loc["RAG"], m.loc["No-RAG"].loc[m.loc["RAG"].index]
        for c in CRIT:
            p, r = wsr(a[c] - b[c])
            rows.append(dict(model=mod, criterion=c, rag=a[c].mean(), norag=b[c].mean(),
                             diff=(a[c] - b[c]).mean(), p=p, r_rb=r))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.values)
    return out


def rag1_vs_rag2(ratings, cas, ground, retr):
    rows = []
    m = per_question(ratings)
    a, b = m.loc["RAG-v2"], m.loc["RAG"].loc[m.loc["RAG-v2"].index]
    for name, c in [("Overall judge score", "overall"), ("Helpfulness", "helpfulness"),
                    ("Justification", "justification")]:
        p, r = wsr(a[c] - b[c]); lo, hi = boot_ci(a[c] - b[c])
        rows.append((name, b[c].mean(), a[c].mean(), p, r, lo, hi))
    cs = cas.set_index(["condition", "question_id"])
    for name, c in [("CAS (all answers)", "CAS"), ("CAS situational relevance", "cas2_relevance")]:
        x, y = cs.loc["RAG-v2", c], cs.loc["RAG", c].loc[cs.loc["RAG-v2", c].index]
        p, r = wsr(x - y); lo, hi = boot_ci(x - y)
        rows.append((name, y.mean(), x.mean(), p, r, lo, hi))
    g = ground.set_index(["condition", "question_id"])
    both = g.loc["RAG1"].join(g.loc["RAG2"], lsuffix="_1", rsuffix="_2")
    both = both[(both.story_used_1 == 1) & (both.story_used_2 == 1)]
    dq = both.parallel_quality_2 - both.parallel_quality_1
    p, r = wsr(dq); lo, hi = boot_ci(dq)
    rows.append((f"Parallel quality (n={len(dq)})", both.parallel_quality_1.mean(),
                 both.parallel_quality_2.mean(), p, r, lo, hi))
    t = retr[retr["C_P@1"].notna()]  # 30 labelled questions
    a0 = t[t.config.str.startswith("A0")].set_index("question_id")["C_P@1"]
    a4 = t[t.config.str.startswith("A4")].set_index("question_id")["C_P@1"].loc[a0.index]
    b01, b10 = int(((a0 == 1) & (a4 == 0)).sum()), int(((a0 == 0) & (a4 == 1)).sum())
    pm = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    rows.append((f"Top-1 right construct (n={len(a0)}; McNemar b={b01}, c={b10})", a0.mean(), a4.mean(), pm,
                 np.nan, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["metric", "RAG1", "RAG2", "p", "r_rb", "ci_lo", "ci_hi"])


if __name__ == "__main__":
    f = sys.argv[1:]
    pd.set_option("display.width", 200)
    print("RAG vs No-RAG (Holm over 4 models x 6 criteria)")
    print(rag_vs_norag(f[0]).round(3).to_string(index=False))
    print("\nRAG1 vs RAG2 (Gemma-2-9B; uncorrected)")
    print(rag1_vs_rag2(pd.read_pickle(f[1]), pd.read_pickle(f[2]), pd.read_pickle(f[3]),
                       pd.read_csv(f[4])).round(3).to_string(index=False))
