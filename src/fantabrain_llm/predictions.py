from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from fantabrain_llm.schema import TrainingExample


class PredictionError(ValueError):
    """Raised when prediction run inputs are inconsistent."""


@dataclass(frozen=True)
class Prediction:
    case_id: int
    mode: str
    task: str
    tags: list[str]
    prompt: str
    expected: str
    prediction: str
    provider: str
    model: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _last_message(example: TrainingExample, role: str) -> str:
    for message in reversed(example.messages):
        if message.role == role:
            return message.content
    raise PredictionError(f"example has no {role} message")


def build_predictions(
    examples: list[TrainingExample],
    responses: Iterable[str],
    provider: str,
    model: str,
) -> list[Prediction]:
    response_list = list(responses)
    if len(response_list) != len(examples):
        raise PredictionError(
            f"expected {len(examples)} responses for {len(examples)} examples, "
            f"got {len(response_list)}"
        )

    predictions: list[Prediction] = []
    for index, (example, response) in enumerate(zip(examples, response_list, strict=True), start=1):
        predictions.append(
            Prediction(
                case_id=index,
                mode=example.mode,
                task=example.task,
                tags=example.tags,
                prompt=_last_message(example, "user"),
                expected=_last_message(example, "assistant"),
                prediction=response,
                provider=provider,
                model=model,
            )
        )
    return predictions


def render_comparison_markdown(predictions: list[Prediction], run_name: str) -> str:
    lines = [
        f"# Prediction Comparison {run_name}",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        "",
    ]

    for prediction in predictions:
        lines.extend(
            [
                f"## Case {prediction.case_id}: {prediction.mode} / {prediction.task}",
                "",
                f"Tags: {', '.join(prediction.tags)}",
                "",
                "**User prompt**",
                "",
                "```text",
                prediction.prompt,
                "```",
                "",
                "**Model prediction**",
                "",
                "```text",
                prediction.prediction,
                "```",
                "",
                "**Expected coach answer**",
                "",
                "```text",
                prediction.expected,
                "```",
                "",
                "**Manual score**",
                "",
                "- Mode correctness: _ / 5",
                "- Tactical usefulness: _ / 5",
                "- Groundedness: _ / 5",
                "- Clarity: _ / 5",
                "- FantaBrain tone: _ / 5",
                "- Notes:",
                "",
            ]
        )

    return "\n".join(lines)


def write_prediction_run(
    predictions: list[Prediction],
    run_name: str,
    eval_path: str,
    output_root: str | Path = "reports/runs",
    metadata: dict[str, object] | None = None,
) -> Path:
    if not predictions:
        raise PredictionError("prediction run must contain at least one prediction")

    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    comparison_path = output_dir / "comparison.md"
    summary_path = output_dir / "summary.json"

    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")

    comparison_path.write_text(
        render_comparison_markdown(predictions, run_name),
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "run_name": run_name,
        "eval_path": eval_path,
        "examples": len(predictions),
        "provider": predictions[0].provider,
        "model": predictions[0].model,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    if metadata:
        summary.update(metadata)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir
