#!/usr/bin/env python3
"""Create clearly labelled provisional AI-simulated ratings for generated answers.

This utility is for pipeline testing and exploratory analysis only. Its output
must never be described as psychologist, clinician, participant, or expert
ratings. It deliberately writes to a directory separate from rater_packets.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_RESPONSES = ROOT.parent / "generated_responses"
DEFAULT_OUTPUT = ROOT / "ai_simulated_provisional_ratings.csv"
RATERS = [f"AI_Simulated_Rater_{index:02d}" for index in range(1, 6)]
METRICS = ["empathy", "relevance", "clarity", "actionability", "justification", "helpfulness"]


def bounded(value: int) -> int:
    return max(1, min(5, value))


def contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def base_scores(response: str, condition: str, retrieved_ids: str) -> dict[str, int]:
    """A transparent, conservative text rubric, not clinical judgement."""
    text = response.strip()
    if not text or text.startswith("ERROR:"):
        return {metric: 1 for metric in METRICS}
    words = re.findall(r"\b\w+\b", text)
    sentences = re.findall(r"[^.!?]+[.!?]", text)
    empathy = 3 + int(contains_any(text, ("i'm sorry", "that sounds", "it makes sense", "understand", "difficult", "painful")))
    relevance = 3 + int(len(words) >= 80) + int(len(words) <= 280)
    clarity = 3 + int(3 <= len(sentences) <= 14) + int("\n" in text or any(marker in text for marker in ("first", "second", "try")))
    actionability = 2 + min(2, sum(marker in text.lower() for marker in ("try ", "consider ", "write ", "talk to", "reach out", "take a")))
    justification = 3
    if condition == "RAG":
        justification += int(bool(retrieved_ids)) + int(contains_any(text, ("passage", "story", "source", "narrative")))
    else:
        justification += int(contains_any(text, ("because", "so that", "this can")))
    helpfulness = round((empathy + relevance + clarity + actionability + justification) / 5)
    return {
        "empathy": bounded(empathy),
        "relevance": bounded(relevance),
        "clarity": bounded(clarity),
        "actionability": bounded(actionability),
        "justification": bounded(justification),
        "helpfulness": bounded(helpfulness),
    }


def small_rater_variation(answer_id: str, rater_id: str, metric: str) -> int:
    digest = hashlib.sha256(f"{answer_id}|{rater_id}|{metric}".encode()).digest()[0]
    return (-1, 0, 0, 0, 1)[digest % 5]


def load_responses(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.csv"))
    if len(files) != 8:
        raise SystemExit(f"Expected 8 generated response CSVs in {directory}, found {len(files)}.")
    frames = [pd.read_csv(file) for file in files]
    responses = pd.concat(frames, ignore_index=True)
    required = {"answer_id", "model", "condition", "question_id", "response_text", "retrieved_story_ids"}
    missing = required.difference(responses.columns)
    if missing:
        raise SystemExit(f"Response CSVs are missing columns: {sorted(missing)}")
    if len(responses) != 320:
        raise SystemExit(f"Expected 320 responses, found {len(responses)}.")
    if responses["answer_id"].duplicated().any():
        raise SystemExit("Duplicate answer IDs found.")
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AI-simulated provisional ratings from generated model responses")
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    responses = load_responses(args.responses_dir)
    rows: list[dict[str, object]] = []
    for _, answer in responses.iterrows():
        baseline = base_scores(str(answer["response_text"]), str(answer["condition"]), str(answer["retrieved_story_ids"] or ""))
        for rater in RATERS:
            scores = {metric: bounded(baseline[metric] + small_rater_variation(str(answer["answer_id"]), rater, metric)) for metric in METRICS}
            rows.append({
                "rating_source": "AI-simulated provisional rating; not a psychologist rating",
                "rater_id": rater,
                "answer_id": answer["answer_id"],
                "model": answer["model"],
                "condition": answer["condition"],
                "question_id": answer["question_id"],
                **scores,
                "feedback_comments": "Automated provisional rubric score. Requires independent human psychologist review.",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} AI-simulated provisional ratings to {args.output}")


if __name__ == "__main__":
    main()
