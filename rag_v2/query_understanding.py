"""Stage 1 - construct detection, risk screening and HyDE-style query enrichment (one LLM call, JSON)."""
from __future__ import annotations
import re
from backends import LLM, parse_json
from constructs import CONSTRUCTS, normalise

PROMPT = """You analyse a message sent to a mental-health support chatbot for Indian young adults.
Return ONLY a JSON object with these keys:
"constructs": 1-2 labels chosen exactly from this list: {labels}
"risk": "high" ONLY if the message mentions or clearly implies self-harm, suicide, wanting to die, or intent to physically harm someone; "low" ONLY if it explicitly expresses giving up on life or that nothing is worth living for; otherwise "none" (ordinary stress, sadness, anger or worry is "none")
"situation": one sentence restating the person's core problem in neutral psychological terms
"hypothetical_episode": 2-3 sentences written like the psychological summary of an episode from an Indian epic (Ramayana, Mahabharata or Bhagavad Gita) in which a character faces the SAME inner struggle. Describe the struggle and the kind of guidance that helps; do not quote verses.

User message: {query}"""

RISK_PATTERNS = re.compile(r"\b(kill myself|suicid|end my life|want to die|self[- ]harm|hurt myself|no reason to live|better off dead)", re.I)


def understand(llm: LLM, query: str) -> dict:
    raw = parse_json(llm.chat([{"role": "user", "content": PROMPT.format(labels=", ".join(CONSTRUCTS), query=query)}], json_mode=True))
    cons = []
    for c in raw.get("constructs", []) or []:
        cons += sorted(normalise(str(c)))
    if not cons:                                  # LLM failed: keyword fallback so the query is never dropped
        from constructs import SYNONYMS
        ql = query.lower()
        cons = [c for c in CONSTRUCTS if any(w in ql for w in SYNONYMS[c])][:2]
    risk = str(raw.get("risk", "none")).lower()
    if RISK_PATTERNS.search(query):
        risk = "high"
    return {"constructs": list(dict.fromkeys(cons))[:2], "risk": risk if risk in {"none", "low", "high"} else "none",
            "situation": str(raw.get("situation", "")), "hypothetical_episode": str(raw.get("hypothetical_episode", ""))}
