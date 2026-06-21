from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.output_filter import (  # noqa: E402
    OutputFilterError,
    filter_prediction_records,
    write_filter_outputs,
)
from fantabrain_llm.prediction_audit import (  # noqa: E402
    PredictionAuditError,
    load_prediction_records,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a FantaBrain prediction run.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the predictions parent directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent

    try:
        records = load_prediction_records(predictions_path)
        report = filter_prediction_records(records)
        json_path, markdown_path = write_filter_outputs(report, output_dir)
    except (OSError, OutputFilterError, PredictionAuditError) as exc:
        print(f"Output filter error: {exc}", file=sys.stderr)
        return 1

    print(f"Output filter JSON written to {json_path}")
    print(f"Output filter Markdown written to {markdown_path}")
    print(f"Cases: {report.cases}")
    print("Decision counts:")
    if report.decision_counts:
        for action, count in report.decision_counts.items():
            print(f"{action}: {count}")
    else:
        print("none: 0")
    print("Violation counts:")
    if report.violation_counts:
        for check, count in report.violation_counts.items():
            print(f"{check}: {count}")
    else:
        print("none: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
