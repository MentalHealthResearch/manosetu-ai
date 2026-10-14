"""Create blinded rating packets from a completed response manifest."""

from __future__ import annotations

import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "response_manifest.csv"
PACKETS = ROOT / "rater_packets"
RATERS = [f"P{i:02d}" for i in range(1, 6)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    responses = read_csv(MANIFEST)
    expected = 4 * 2 * 40
    if len(responses) != expected:
        raise ValueError(f"Expected {expected} response rows, found {len(responses)}.")
    missing = [row["answer_id"] for row in responses if not row["response_text"].strip()]
    if missing:
        raise ValueError(f"Response text is missing for {len(missing)} answers. Example: {missing[0]}")

    indexed = sorted(responses, key=lambda row: row["answer_id"])
    private_key = []
    for index, row in enumerate(indexed, start=1):
        private_key.append({
            "blind_answer_id": f"A{index:04d}",
            "answer_id": row["answer_id"],
            "model": row["model"],
            "condition": row["condition"],
            "question_id": row["question_id"],
        })

    write_csv(
        PACKETS / "private_assignment_key.csv",
        private_key,
        ["blind_answer_id", "answer_id", "model", "condition", "question_id"],
    )

    answers = {row["answer_id"]: row for row in responses}
    for rater_index, rater_id in enumerate(RATERS, start=1):
        rows = []
        for key in private_key:
            answer = answers[key["answer_id"]]
            rows.append({
                "rater_id": rater_id,
                "blind_answer_id": key["blind_answer_id"],
                "question_id": answer["question_id"],
                "question_text": answer["question_text"],
                "response_text": answer["response_text"],
                "empathy": "",
                "relevance": "",
                "clarity": "",
                "actionability": "",
                "justification": "",
                "helpfulness": "",
                "feedback_comments": "",
            })
        random.Random(20260924 + rater_index).shuffle(rows)
        write_csv(
            PACKETS / f"psychologist_{rater_index:02d}_rating_form.csv",
            rows,
            list(rows[0]),
        )

    print(f"Created {len(RATERS)} blinded packets with {len(private_key)} answers each.")


if __name__ == "__main__":
    main()
