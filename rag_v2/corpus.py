"""Load the Ramayanam / Gita / Mahabharat JSON corpus with the same story IDs as ingest_all_narrative_corpus.py."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from constructs import normalise_many

CORPORA = ("Ramayanam", "Gita", "Mahabharat")
SOURCE_NAME = {"ramayanam": "Valmiki Ramayana", "gita": "Bhagavad Gita", "mahabharat": "Mahabharata"}


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(compact(v) for v in value if compact(v))
    if isinstance(value, dict):
        return "; ".join(f"{k}: {compact(v)}" for k, v in value.items() if compact(v))
    return str(value)


def _objects(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        dec, pos, out = json.JSONDecoder(), 0, []
        while pos < len(text):
            while pos < len(text) and text[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(text):
                break
            obj, pos = dec.raw_decode(text, pos)
            out.append(obj)
        return out


def load_corpus(root: Path) -> list[dict]:
    records = []
    for corpus in CORPORA:
        base = root / corpus
        for f in sorted(base.rglob("*.json")):
            objs = _objects(f)
            rel = f.relative_to(base).with_suffix("").as_posix()
            src = corpus.lower()
            for i, raw in enumerate(objs, 1):
                suffix = "" if len(objs) == 1 else f"#part{i}"
                source_text = raw.get("Source_Text", "") or ""
                if not source_text or ".csv" in source_text:
                    source_text = SOURCE_NAME[src]
                r = {
                    "story_id": f"{src}:{rel}{suffix}", "corpus": src, "source_text": source_text,
                    "reference": compact(raw.get("Chapter/Canto/Verse_Reference")),
                    "characters": [c.strip() for c in (raw.get("Characters_Involved") or []) if isinstance(c, str)],
                    "paragraph": compact(raw.get("Paragraph", raw.get("paragraph", ""))),
                    "situation": compact(raw.get("Situation/Problem_Faced")),
                    "teaching": compact(raw.get("Key_Teaching/Action_Taken")),
                    "outcome": compact(raw.get("Outcome/Resolution")),
                    "themes_raw": raw.get("Mental_Health_Themes") or [],
                    "synonyms": compact(raw.get("Mental_Health_Theme_Synonyms")),
                    "psych_summary": compact(raw.get("Brief_Summary_Psychological")),
                    "spiritual_summary": compact(raw.get("Brief_Summary_Spiritual")),
                    "concepts": compact(raw.get("Spiritual_Psychological_Concept_Indian")),
                }
                r["constructs"] = sorted(normalise_many(r["themes_raw"]))
                records.append(r)
    return records


def retrieval_text(r: dict) -> str:
    """Identical field composition to the baseline ingester (keeps A0 comparable)."""
    fields = [("Corpus", r["corpus"]), ("Source", r["source_text"]), ("Reference", r["reference"]),
              ("Characters", ", ".join(r["characters"])), ("Situation", r["situation"]), ("Teaching", r["teaching"]),
              ("Outcome", r["outcome"]), ("Mental health themes", compact(r["themes_raw"])),
              ("Theme synonyms", r["synonyms"]), ("Psychological summary", r["psych_summary"]),
              ("Spiritual summary", r["spiritual_summary"]), ("Concepts", r["concepts"]), ("Passage", r["paragraph"][:1000])]
    return "\n".join(f"{k}: {v}" for k, v in fields if v)


def rerank_text(r: dict) -> str:
    """Compact, psychologically focused view of an episode for the cross-encoder."""
    return (f"{r['source_text']} {r['reference']}. Characters: {', '.join(r['characters'])}. "
            f"Situation: {r['situation']} Teaching: {r['teaching']} Psychological summary: {r['psych_summary']} "
            f"Themes: {compact(r['themes_raw'])}")
