from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.scoring import ScoreError, aggregate_scores, load_scores_csv  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply manual P1 scores to a prediction run.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl.")
    parser.add_argument("--scores", required=True, help="Path to scores CSV.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the predictions parent directory.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ScoreError(f"Summary must be a JSON object: {path}")
    return payload


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ScoreError(f"Predictions file not found: {path}")

    predictions: list[dict[str, Any]] = []
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
            predictions.append(payload)

    if not predictions:
        raise ScoreError(f"Predictions file has no rows: {path}")
    return predictions


def _validate_case_alignment(predictions: list[dict[str, Any]], score_case_ids: list[int]) -> None:
    prediction_case_ids = [int(prediction["case_id"]) for prediction in predictions]
    if prediction_case_ids != score_case_ids:
        raise ScoreError(
            "scores case ids must match predictions case ids in order: "
            f"predictions={prediction_case_ids}, scores={score_case_ids}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent

    try:
        predictions = _load_predictions(predictions_path)
        scores = load_scores_csv(args.scores)
        _validate_case_alignment(predictions, [row.case_id for row in scores])

        summary = _load_json(output_dir / "summary.json")
        summary["scoring"] = aggregate_scores(scores)
        if "run_name" not in summary:
            summary["run_name"] = output_dir.name

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "scores.json").write_text(
            json.dumps([row.to_dict() for row in scores], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "summary.scored.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except (KeyError, ScoreError) as exc:
        print(f"Score error: {exc}", file=sys.stderr)
        return 1

    print(f"Scored summary written to {output_dir / 'summary.scored.json'}")
    print(f"Scores JSON written to {output_dir / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
