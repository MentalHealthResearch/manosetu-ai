#!/usr/bin/env python3
"""Full RAG-v2 generation for the 40 benchmark questions.

Pipeline per question:
  1 query understanding (constructs, risk, HyDE)      4 construct-aware grounded prompt with ONE episode
  2 hybrid retrieval + construct boost (A4)             (or no-story answer when retrieval abstains)
  3 cross-encoder rerank + abstention threshold (A5)  5 faithfulness check -> one regeneration if unsupported
  + safety gate: risk=high/low adds explicit safety check-in and crisis signposting.
Output: outputs/generated_responses_v2/<Model>__RAG-v2.csv in the benchmark schema (+ audit columns),
so the same AI judges / CAS scorer / psychologist packets can be applied unchanged.
"""
from __future__ import annotations
import argparse, time
from datetime import datetime, timezone
import pandas as pd
from common import CFG, OUT, path, load_qu_cache, save_qu_cache
from corpus import load_corpus
from backends import Encoder, Reranker, LLM, parse_json
from query_understanding import understand
from retriever import HybridRetriever, CONFIGS

PERSONA = ("You are a supportive mental-health conversation assistant for young adults in India. Provide culturally "
           "sensitive, non-judgmental support. Validate the person's feelings and give 2-3 practical, specific next steps. "
           "Do not diagnose.")

STORY_RULES = """Use the ONE narrative episode below as a short illustration.
- Retell only details present in the episode fields; do not add characters, dialogue, quotations or verse numbers that are not there.
- Name the source and the characters naturally (e.g. "In the Ramayana, when ..."). Never mention 'passages', 'sources' or IDs.
- Make the parallel to the person's situation explicit, then move to practical steps. Keep the story to 2-4 sentences."""

PARAMETRIC = ("You may illustrate your answer with ONE relevant episode from the Ramayana, Mahabharata or Bhagavad Gita "
              "from your own knowledge (2-4 sentences), making the parallel explicit, then give practical steps.")

NO_STORY = "No sufficiently relevant narrative episode was found, so do not use any mythological or scriptural reference."

SAFETY = ("The message contains hopelessness or possible risk cues. Gently and directly check on their safety, say that "
          "support is available, and encourage contacting a trusted person or a crisis line (in India, Tele-MANAS 14416) "
          "or local emergency services if they feel unsafe.")

FAITH = """You check whether the STORY PART of a chatbot reply is faithful to a source episode.
Episode fields:
{episode}

Reply:
{reply}

Step 1: consider ONLY sentences that mention a character, an epic or scripture (Ramayana, Mahabharata, Gita), a quotation, or an event from a story.
IGNORE empathy, advice, suggestions, lists of steps and general statements about the user - they are never unsupported claims.
Step 2: for those story sentences only, list details that contradict the episode or are NOT present in it (e.g. a different character, an invented event, quote or verse).
Paraphrase and brief comparisons to the user are fine.
Return ONLY JSON: {{"faithful": true/false, "unsupported_claims": ["exact story detail that is unsupported", ...]}}"""

EPIC_WORDS = ("rama", "sita", "seetha", "lakshmana", "hanuman", "arjuna", "krishna", "yudhi", "bhishma", "sugr", "vali",
              "dasharatha", "dasaratha", "bharata", "kausalya", "kaikeyi", "ravana", "draupadi", "karna", "vyasa", "jambavan",
              "angada", "sumantra", "vishvamitra", "visvamitra", "gita", "ramayana", "mahabharata", "kurukshetra", "ayodhya",
              "pandava", "kaurava", "duryodhana", "bhima", "drona", "vidura", "shloka", "verse")


def story_claims_only(claims, ep):
    """Keep only verifier claims that are actually about the narrative (guards against over-flagging advice)."""
    names = {c.lower() for c in ep["characters"]}
    keep = []
    for c in claims or []:
        t = str(c).lower()
        if any(w in t for w in EPIC_WORDS) or any(n and n in t for n in names):
            keep.append(str(c))
    return keep


import re
RISKY_EP = re.compile(r"(end (his|her|my) life|take (his|her) (own )?life|suicid|wish(ed)? to die|give up (his|her) life|"
                      r"continu\w* (to live|existence)|no reason to live|kill (him|her)self|self-destruct|"
                      r"purpose\w* of (my|his|her) life|meaning (and purpose )?of (continued )?(life|existence))", re.I)


def episode_block(r: dict) -> str:
    return (f"Source: {r['source_text']} ({r['reference']})\nCharacters: {', '.join(r['characters'])}\n"
            f"Situation: {r['situation']}\nKey teaching / action: {r['teaching']}\nOutcome: {r['outcome']}\n"
            f"Psychological summary: {r['psych_summary']}\nPassage (excerpt): {r['paragraph'][:1200]}")


# Prompt versions:
#   2.1 = version used for the evaluated RAG2 and memory-control outputs (safety block only when query risk is low/high)
#   2.2 = post-evaluation revision (safety block also on Feelings of Hopelessness or an episode describing despair
#         about living, plus the "do not echo" instruction). Not evaluated.
PROMPT_VERSION = "2.2"


def build_prompt(q: str, qu: dict, ep: dict | None, parametric: bool = False) -> str:
    parts = [PERSONA]
    if qu["constructs"]:
        parts.append(f"The main concern appears to be: {', '.join(qu['constructs'])}. Tailor the steps to it.")
    if PROMPT_VERSION == "2.1":
        risky_episode = False
        needs_safety = qu["risk"] in {"low", "high"}
    else:
        risky_episode = bool(ep) and bool(RISKY_EP.search(" ".join([ep["situation"], ep["teaching"], ep["psych_summary"], ep["paragraph"]])))
        needs_safety = qu["risk"] != "none" or "Feelings of Hopelessness" in qu["constructs"] or risky_episode
    if needs_safety:
        parts.append(SAFETY)
    if risky_episode:
        parts.append("The episode touches on despair about living. Do NOT echo thoughts of not wanting to live; frame the "
                     "story only around how the character was supported and recovered.")
    parts.append(PARAMETRIC if parametric else (STORY_RULES + "\n\nEpisode:\n" + episode_block(ep) if ep else NO_STORY))
    parts.append(f"User message: {q}\n\nWrite a warm, clear response of at most {CFG['max_words']} words.")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", action="append", help="label from config generation_models")
    ap.add_argument("--parametric", action="store_true",
                    help="control arm: same pipeline minus retrieval; model may use stories from its own knowledge")
    ap.add_argument("--prompt-version", choices=["2.1", "2.2"], default="2.2",
                    help="2.1 reproduces the evaluated outputs; 2.2 is the post-evaluation safety revision")
    a = ap.parse_args()
    global PROMPT_VERSION
    PROMPT_VERSION = a.prompt_version
    cond = "NoRAG-narrative" if a.parametric else "RAG-v2"
    recs = load_corpus(path("dataset_root"))
    qs = pd.read_csv(path("questions_csv"))
    judge = LLM(CFG["llm_model"], 0.0, a.mock)
    cache = load_qu_cache(a.mock)
    conf = CONFIGS[CFG.get("generation_retrieval_config", "A4_+construct_boost")]
    ret = None if a.parametric else HybridRetriever(
        recs, {"main": Encoder(CFG["embedding_model"], a.mock)},
        Reranker(CFG["reranker_model"], a.mock) if conf["rerank"] else None, CFG)
    out_dir = OUT / ("generated_responses_v2_mock" if a.mock else "generated_responses_v2"); out_dir.mkdir(exist_ok=True)

    for label, tag in CFG["generation_models"].items():
        if a.model and label not in a.model:
            continue
        gen = LLM(tag, CFG["temperature"], a.mock)
        suffix = "" if PROMPT_VERSION == "2.1" else f"__v{PROMPT_VERSION}"   # never resume across versions
        dest = out_dir / f"{label}__{cond}{suffix}.csv"
        done = pd.read_csv(dest) if dest.exists() else pd.DataFrame()
        rows = done.to_dict("records"); have = {int(x["question_id"]) for x in rows}
        for r in qs.itertuples():
            qid = int(r.question_id)
            if qid in have:
                continue
            q = r.question_text
            if str(qid) not in cache:
                cache[str(qid)] = understand(judge, q); save_qu_cache(cache, a.mock)
            qu = cache[str(qid)]
            if a.parametric:
                res, use, ep = [], False, None
            else:
                res = ret.search(q, qu, conf)
                top = res[0]
                if conf["rerank"]:
                    use = top.get("rerank", 1.0) >= CFG["abstain_threshold"]
                else:   # abstain when the best episode shares no construct with the detected concern
                    use = (not qu["constructs"]) or bool(set(qu["constructs"]) & set(recs[top["idx"]]["constructs"]))
                ep = recs[top["idx"]] if use else None
            t0 = time.perf_counter()
            reply = gen.chat([{"role": "user", "content": build_prompt(q, qu, ep, a.parametric)}]).strip()
            faithful_first = faithful_final = None; unsupported = []; regenerated = False
            if ep is not None:
                v = parse_json(judge.chat([{"role": "user", "content": FAITH.format(episode=episode_block(ep), reply=reply)}], json_mode=True))
                unsupported = story_claims_only(v.get("unsupported_claims", []), ep)
                faithful_first = faithful_final = not unsupported
                if not faithful_first and unsupported:
                    fix = build_prompt(q, qu, ep) + "\n\nYour previous draft contained details not in the episode: " + \
                          "; ".join(map(str, unsupported)) + ". Rewrite the response without them."
                    reply = gen.chat([{"role": "user", "content": fix}]).strip(); regenerated = True
                    v2 = parse_json(judge.chat([{"role": "user", "content": FAITH.format(episode=episode_block(ep), reply=reply)}], json_mode=True))
                    faithful_final = not story_claims_only(v2.get("unsupported_claims", []), ep)
            rows.append({
                "answer_id": f"{label}__{cond}__Q{qid:02d}", "model": label, "model_tag": tag, "condition": cond,
                "question_id": qid, "question_text": q, "response_text": reply,
                "retrieved_story_ids": ";".join(c["story_id"] for c in res[:CFG["final_k"]]),
                "retrieval_scores": ";".join(f"{c.get('rerank', c['fused']):.4f}" for c in res[:CFG["final_k"]]),
                "used_episode": ep["story_id"] if ep else "", "abstained": (not use) and not a.parametric,
                "constructs_detected": ";".join(qu["constructs"]), "risk": qu["risk"],
                "faithful_first_draft": faithful_first, "faithful_final": faithful_final, "regenerated": regenerated,
                "unsupported_claims_first_draft": " | ".join(map(str, unsupported)),
                "prompt_version": ("rag_v2_parametric" if PROMPT_VERSION == "2.1" else f"rag_v{PROMPT_VERSION}_parametric") if a.parametric else f"rag_v{PROMPT_VERSION}", "retrieval_config": "" if a.parametric else CFG.get("generation_retrieval_config"), "generation_seconds": round(time.perf_counter() - t0, 2),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            pd.DataFrame(rows).sort_values("question_id").to_csv(dest, index=False)
            print(f"{label} Q{qid:02d} constructs={qu['constructs']} risk={qu['risk']} episode={'-' if not ep else ep['story_id']} "
                  f"faithful={faithful_first}->{faithful_final}", flush=True)
        print("Saved", dest)


if __name__ == "__main__":
    main()
