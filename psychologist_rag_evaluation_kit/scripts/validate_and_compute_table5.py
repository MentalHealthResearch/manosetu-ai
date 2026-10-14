"""Validate five blinded packets and calculate the Table 5 means."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKETS = ROOT / "rater_packets"
COMPLETED = PACKETS / "completed"
OUTPUTS = ROOT / "outputs" / "f82004b3"
MODELS = ["Llama-3.1-8B", "Mistral-7B", "Gemma-2-9B", "MentalLLaMA-13B"]
CONDITIONS = ["RAG", "No-RAG"]
METRICS = ["empathy", "relevance", "clarity", "actionability", "justification", "helpfulness"]
EXPECTED_ANSWERS = 320
EXPECTED_RATERS = 5
EXPECTED_PER_CELL = 40 * EXPECTED_RATERS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_score(value: str, location: str) -> int:
    try:
        score = int(value)
    except ValueError as exc:
        raise ValueError(f"{location} must contain an integer 1-5, got {value!r}.") from exc
    if score not in range(1, 6):
        raise ValueError(f"{location} must contain an integer 1-5, got {value!r}.")
    return score


def main() -> None:
    key_rows = read_csv(PACKETS / "private_assignment_key.csv")
    if len(key_rows) != EXPECTED_ANSWERS:
        raise ValueError("The private assignment key must contain 320 answers.")
    key = {row["blind_answer_id"]: row for row in key_rows}
    if len(key) != EXPECTED_ANSWERS:
        raise ValueError("Duplicate blind_answer_id values in the assignment key.")

    values: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for index in range(1, EXPECTED_RATERS + 1):
        path = COMPLETED / f"psychologist_{index:02d}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed packet: {path}")
        rows = read_csv(path)
        if len(rows) != EXPECTED_ANSWERS:
            raise ValueError(f"{path.name} must contain {EXPECTED_ANSWERS} answers, found {len(rows)}.")
        ids = [row["blind_answer_id"] for row in rows]
        if set(ids) != set(key) or len(ids) != len(set(ids)):
            raise ValueError(f"{path.name} has missing, unknown, or duplicate blind answer IDs.")
        rater_ids = {row["rater_id"] for row in rows}
        if rater_ids != {f"P{index:02d}"}:
            raise ValueError(f"{path.name} must contain only rater_id P{index:02d}.")
        for row in rows:
            blind_id = row["blind_answer_id"]
            if (row["rater_id"], blind_id) in seen:
                raise ValueError(f"Duplicate rating for {row['rater_id']} / {blind_id}.")
            seen.add((row["rater_id"], blind_id))
            mapped = key[blind_id]
            for metric in METRICS:
                values[(mapped["model"], mapped["condition"], metric)].append(
                    require_score(row[metric].strip(), f"{path.name} {blind_id} {metric}")
                )

    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            output = {"model": model, "condition": condition, "ratings_per_metric": EXPECTED_PER_CELL}
            for metric in METRICS:
                metric_values = values[(model, condition, metric)]
                if len(metric_values) != EXPECTED_PER_CELL:
                    raise ValueError(f"{model} {condition} {metric} has {len(metric_values)} scores.")
                output[metric] = round(sum(metric_values) / len(metric_values), 2)
            rows.append(output)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fields = ["model", "condition", "empathy", "relevance", "clarity", "actionability", "justification", "helpfulness", "ratings_per_metric"]
    with (OUTPUTS / "table5_means.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Table 5: Mean expert ratings (1-5)",
        "",
        "| Model | Cond. | Empathy | Relevance | Clarity | Action. | Justif. | Helpfulness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['empathy']:.2f} | {row['relevance']:.2f} | "
            f"{row['clarity']:.2f} | {row['actionability']:.2f} | {row['justification']:.2f} | {row['helpfulness']:.2f} |"
        )
    (OUTPUTS / "table5_means.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Validated all five packets and wrote Table 5 means.")


if __name__ == "__main__":
    main()
