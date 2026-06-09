from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.prediction_audit import (  # noqa: E402
    PredictionAuditError,
    audit_prediction_records,
    load_prediction_records,
    write_audit_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a FantaBrain prediction run.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the predictions parent directory.",
    )
    parser.add_argument(
        "--fail-on-hard-gates",
        action="store_true",
        help="Exit non-zero when hard audit violations are found.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent

    try:
        records = load_prediction_records(predictions_path)
        report = audit_prediction_records(records)
        json_path, markdown_path = write_audit_outputs(report, output_dir)
    except PredictionAuditError as exc:
        print(f"Prediction audit error: {exc}", file=sys.stderr)
        return 1

    print(f"Prediction audit JSON written to {json_path}")
    print(f"Prediction audit Markdown written to {markdown_path}")
    print(f"Cases: {report.cases}")
    print(f"Hard violations: {report.hard_violation_count}")
    for check, count in report.summary.items():
        print(f"{check}: {count}")

    if args.fail_on_hard_gates and report.hard_violation_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
