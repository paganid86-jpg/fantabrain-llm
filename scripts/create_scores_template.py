from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.scoring import REQUIRED_COLUMNS, ScoreError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a blank manual scoring CSV template.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl.")
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path. Defaults to scores.template.csv next to predictions.",
    )
    return parser.parse_args(argv)


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ScoreError(f"Predictions file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoreError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ScoreError(f"{path}:{line_number}: prediction must be a JSON object")
            rows.append(payload)
    if not rows:
        raise ScoreError(f"Predictions file has no rows: {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions_path = Path(args.predictions)
    output_path = (
        Path(args.output)
        if args.output
        else predictions_path.parent / "scores.template.csv"
    )

    try:
        predictions = _load_predictions(predictions_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            for prediction in predictions:
                writer.writerow(
                    {
                        "case": prediction["case_id"],
                        "mode": "",
                        "tactical": "",
                        "grounded": "",
                        "clarity": "",
                        "tone": "",
                        "hallucination_free": "",
                        "notes": "",
                    }
                )
    except (KeyError, ScoreError) as exc:
        print(f"Template error: {exc}", file=sys.stderr)
        return 1

    print(f"Scores template written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
