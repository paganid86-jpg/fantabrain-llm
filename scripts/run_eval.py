from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.dataset import DatasetError, load_examples  # noqa: E402
from fantabrain_llm.prompts import MANUAL_REVIEW_RUBRIC  # noqa: E402
from fantabrain_llm.schema import TrainingExample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a manual eval report for a pagella set.")
    parser.add_argument("--eval", required=True, help="Eval JSONL path.")
    parser.add_argument("--run-name", required=True, help="Name for reports/runs/<run-name>.")
    parser.add_argument("--output-root", default="reports/runs", help="Report output directory.")
    return parser.parse_args()


def _last_message(example: TrainingExample, role: str) -> str:
    for message in reversed(example.messages):
        if message.role == role:
            return message.content
    return ""


def render_markdown(examples: list[TrainingExample], run_name: str) -> str:
    lines = [
        f"# Pagella {run_name}",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        "",
        "## Rubric",
        "",
    ]
    lines.extend(f"- [ ] {item}" for item in MANUAL_REVIEW_RUBRIC)
    lines.append("")

    for index, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"## Case {index}: {example.mode} / {example.task}",
                "",
                "**User prompt**",
                "",
                "```text",
                _last_message(example, "user"),
                "```",
                "",
                "**Expected coach answer**",
                "",
                "```text",
                _last_message(example, "assistant"),
                "```",
                "",
                "**Manual score**",
                "",
                "- Mode correctness: _ / 5",
                "- Tactical usefulness: _ / 5",
                "- Groundedness: _ / 5",
                "- FantaBrain tone: _ / 5",
                "- Notes:",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        examples = load_examples(args.eval)
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_root) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "manual_review.md"
    summary_path = output_dir / "summary.json"

    markdown_path.write_text(render_markdown(examples, args.run_name), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "eval_path": args.eval,
                "examples": len(examples),
                "generated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Pagella written to {markdown_path}")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
