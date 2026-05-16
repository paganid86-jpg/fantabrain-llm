from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ScoreError(ValueError):
    """Raised when manual scoring input is missing or inconsistent."""


SCORE_FIELDS = ("mode", "tactical", "grounded", "clarity", "tone")
REQUIRED_COLUMNS = ("case", *SCORE_FIELDS, "hallucination_free", "notes")


@dataclass(frozen=True)
class ScoreRow:
    case_id: int
    mode: int
    tactical: int
    grounded: int
    clarity: int
    tone: int
    hallucination_free: bool
    notes: str = ""

    @property
    def raw_average(self) -> float:
        return sum(getattr(self, field) for field in SCORE_FIELDS) / len(SCORE_FIELDS)

    @property
    def effective_average(self) -> float:
        if self.hallucination_free:
            return self.raw_average
        return min(self.raw_average, 1.0)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["raw_average"] = round(self.raw_average, 3)
        payload["effective_average"] = round(self.effective_average, 3)
        return payload


def _require_int(value: str | None, field_name: str, line_number: int) -> int:
    if value is None or not value.strip():
        raise ScoreError(f"line {line_number}: {field_name} is required")
    try:
        return int(value)
    except ValueError as exc:
        raise ScoreError(f"line {line_number}: {field_name} must be an integer") from exc


def _score_value(value: str | None, field_name: str, line_number: int) -> int:
    score = _require_int(value, field_name, line_number)
    if not 1 <= score <= 5:
        raise ScoreError(f"line {line_number}: {field_name} must be between 1 and 5")
    return score


def _hallucination_flag(value: str | None, line_number: int) -> bool:
    flag = _require_int(value, "hallucination_free", line_number)
    if flag not in (0, 1):
        raise ScoreError(f"line {line_number}: hallucination_free must be 0 or 1")
    return bool(flag)


def score_row_from_csv(row: dict[str, str], line_number: int) -> ScoreRow:
    missing = [column for column in REQUIRED_COLUMNS if column not in row]
    if missing:
        raise ScoreError(f"line {line_number}: missing columns: {', '.join(missing)}")

    return ScoreRow(
        case_id=_require_int(row.get("case"), "case", line_number),
        mode=_score_value(row.get("mode"), "mode", line_number),
        tactical=_score_value(row.get("tactical"), "tactical", line_number),
        grounded=_score_value(row.get("grounded"), "grounded", line_number),
        clarity=_score_value(row.get("clarity"), "clarity", line_number),
        tone=_score_value(row.get("tone"), "tone", line_number),
        hallucination_free=_hallucination_flag(row.get("hallucination_free"), line_number),
        notes=(row.get("notes") or "").strip(),
    )


def load_scores_csv(path: str | Path) -> list[ScoreRow]:
    source = Path(path)
    if not source.exists():
        raise ScoreError(f"Scores file not found: {source}")

    rows: list[ScoreRow] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ScoreError(f"Scores file has no header: {source}")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ScoreError(f"Scores file missing columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            rows.append(score_row_from_csv(row, line_number))

    if not rows:
        raise ScoreError(f"Scores file has no rows: {source}")
    return rows


def aggregate_scores(rows: list[ScoreRow]) -> dict[str, Any]:
    if not rows:
        raise ScoreError("cannot aggregate zero score rows")

    cases = len(rows)
    hallucination_free_count = sum(1 for row in rows if row.hallucination_free)
    capped_cases = cases - hallucination_free_count
    field_averages = {
        field: round(sum(getattr(row, field) for row in rows) / cases, 3)
        for field in SCORE_FIELDS
    }

    return {
        "cases": cases,
        "averages": field_averages,
        "raw_average": round(sum(row.raw_average for row in rows) / cases, 3),
        "effective_average": round(sum(row.effective_average for row in rows) / cases, 3),
        "hallucination_free_count": hallucination_free_count,
        "hallucination_free_rate": round(hallucination_free_count / cases, 3),
        "capped_cases": capped_cases,
    }
