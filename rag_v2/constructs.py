"""The 13 study constructs, synonym lists and normalisation of corpus theme labels."""
from __future__ import annotations

CONSTRUCTS = [
    "Stress Management", "Anxiety", "Fear", "Low Mood", "Sadness", "Feelings of Hopelessness",
    "Anger Management", "Emotional Regulation", "Relationship Challenges", "Ego", "Self-Esteem",
    "Work-Life Balance", "Loneliness",
]

SYNONYMS = {
    "Stress Management": ["stress", "pressure", "overwhelmed", "burden", "exam", "coping", "tension"],
    "Anxiety": ["anxiety", "nervous", "worry", "panic", "judged", "performance", "restless"],
    "Fear": ["fear", "afraid", "scared", "avoidance", "courage", "dread", "risk"],
    "Low Mood": ["low mood", "demotivated", "apathy", "listless", "dejected", "despondent", "no interest"],
    "Sadness": ["sadness", "grief", "sorrow", "loss", "separation", "mourning", "longing", "homesick"],
    "Feelings of Hopelessness": ["hopelessness", "despair", "futility", "give up", "pointless", "meaningless"],
    "Anger Management": ["anger", "rage", "fury", "revenge", "outburst", "provoked", "resentment"],
    "Emotional Regulation": ["mood swings", "equanimity", "composure", "stability", "overwhelmed", "self-control"],
    "Relationship Challenges": ["family", "parents", "sibling", "partner", "conflict", "betrayal", "friendship", "duty"],
    "Ego": ["ego", "pride", "arrogance", "criticism", "humility", "correction", "identity"],
    "Self-Esteem": ["self-esteem", "worthless", "failure", "not good enough", "confidence", "comparison", "self-doubt"],
    "Work-Life Balance": ["work-life balance", "burnout", "exhaustion", "rest", "overwork", "duty", "renunciation"],
    "Loneliness": ["loneliness", "alone", "isolation", "exile", "companionship", "friendless", "outsider"],
}

_RULES = [
    ("relationship", "Relationship Challenges"), ("work-life", "Work-Life Balance"), ("work life", "Work-Life Balance"),
    ("self-esteem", "Self-Esteem"), ("self esteem", "Self-Esteem"), ("hopeless", "Feelings of Hopelessness"),
    ("low mood", "Low Mood"), ("depress", "Low Mood"), ("lonel", "Loneliness"), ("isolation", "Loneliness"),
    ("anger", "Anger Management"), ("rage", "Anger Management"), ("emotional regulation", "Emotional Regulation"),
    ("stress", "Stress Management"), ("anxi", "Anxiety"), ("fear", "Fear"), ("sadness", "Sadness"),
    ("grief", "Sadness"), ("ego", "Ego"),
]


def normalise(label: str) -> set[str]:
    """Map a free-text theme label (corpus or LLM output) to the canonical construct set."""
    text = (label or "").strip().lower()
    for canon in CONSTRUCTS:
        if text == canon.lower():
            return {canon}
    return {canon for key, canon in _RULES if key in text}


def normalise_many(labels) -> set[str]:
    out: set[str] = set()
    for label in labels or []:
        out |= normalise(str(label))
    return out
