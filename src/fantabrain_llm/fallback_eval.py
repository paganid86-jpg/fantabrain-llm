from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from fantabrain_llm.openai_fallback import FallbackResponse, FallbackUsage
from fantabrain_llm.output_filter import FilterAction, filter_model_output

SAFE_FALLBACK_RESPONSE = (
    "Damn, non ho abbastanza contesto per chiuderla con sicurezza.\n"
    "Ti do una lettura prudente: evita scelte basate su dati non confermati e confronta "
    "titolarita, copertura, modalita e rischio prima di decidere."
)


class FallbackEvalError(ValueError):
    """Raised when fallback eval inputs are invalid."""


class FinalSource(StrEnum):
    PRIMARY = "primary"
    FALLBACK = "fallback"
    SAFE = "safe"


class FallbackClient(Protocol):
    def generate(self, *, mode: str, task: str, prompt: str) -> FallbackResponse:
        """Generate a fallback response for a failed primary model output."""


@dataclass(frozen=True)
class FallbackCaseResult:
    case_id: int
    mode: str
    task: str
    prompt: str
    expected: str | None
    primary_prediction: str
    primary_action: str
    primary_reason: str
    primary_violations: list[dict[str, object]]
    fallback_used: bool
    fallback_prediction: str | None
    fallback_action: str | None
    fallback_reason: str | None
    fallback_violations: list[dict[str, object]]
    final_prediction: str
    final_source: FinalSource
    usage: FallbackUsage | None
    estimated_cost_usd: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "task": self.task,
            "prompt": self.prompt,
            "expected": self.expected,
            "primary_prediction": self.primary_prediction,
            "primary_action": self.primary_action,
            "primary_reason": self.primary_reason,
            "primary_violations": self.primary_violations,
            "fallback_used": self.fallback_used,
            "fallback_prediction": self.fallback_prediction,
            "fallback_action": self.fallback_action,
            "fallback_reason": self.fallback_reason,
            "fallback_violations": self.fallback_violations,
            "final_prediction": self.final_prediction,
            "final_source": self.final_source.value,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass(frozen=True)
class FallbackEvalReport:
    cases: int
    results: list[FallbackCaseResult]
    primary_action_counts: dict[str, int]
    fallback_used_count: int
    fallback_success_count: int
    final_source_counts: dict[str, int]
    unresolved_safe_count: int
    primary_violation_counts: dict[str, int]
    final_violation_counts: dict[str, int]
    estimated_total_cost_usd: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "cases": self.cases,
            "primary_action_counts": self.primary_action_counts,
            "fallback_used_count": self.fallback_used_count,
            "fallback_success_count": self.fallback_success_count,
            "final_source_counts": self.final_source_counts,
            "unresolved_safe_count": self.unresolved_safe_count,
            "primary_violation_counts": self.primary_violation_counts,
            "final_violation_counts": self.final_violation_counts,
            "estimated_total_cost_usd": self.estimated_total_cost_usd,
            "results": [result.to_dict() for result in self.results],
        }


def run_fallback_eval(
    records: list[dict[str, object]],
    *,
    fallback_client: FallbackClient,
) -> FallbackEvalReport:
    results: list[FallbackCaseResult] = []

    for record in records:
        case_id = _require_int(record, "case_id")
        mode = _require_str(record, "mode")
        task = _require_str(record, "task")
        prompt = _require_str(record, "prompt")
        expected = _require_expected(record)
        primary_prediction = _require_prediction(record)

        primary_decision = filter_model_output(
            mode=mode,
            task=task,
            prompt=prompt,
            prediction=primary_prediction,
            case_id=case_id,
        )
        primary_violations = _violation_dicts(primary_decision.violations)

        fallback_used = primary_decision.action in {
            FilterAction.FALLBACK,
            FilterAction.SAFE,
        }
        fallback_prediction: str | None = None
        fallback_action: str | None = None
        fallback_reason: str | None = None
        fallback_violations: list[dict[str, object]] = []
        usage: FallbackUsage | None = None
        estimated_cost_usd: float | None = None

        if fallback_used:
            fallback_response = fallback_client.generate(
                mode=mode,
                task=task,
                prompt=prompt,
            )
            fallback_prediction = fallback_response.text
            usage = fallback_response.usage
            estimated_cost_usd = usage.estimated_cost_usd

            fallback_decision = filter_model_output(
                mode=mode,
                task=task,
                prompt=prompt,
                prediction=fallback_prediction,
                fallback_failed=True,
                case_id=case_id,
            )
            fallback_action = fallback_decision.action.value
            fallback_reason = fallback_decision.reason
            fallback_violations = _violation_dicts(fallback_decision.violations)

            if fallback_decision.action in {
                FilterAction.PASS,
                FilterAction.PASS_WITH_WARNINGS,
            }:
                final_source = FinalSource.FALLBACK
                final_prediction = fallback_prediction
            else:
                final_source = FinalSource.SAFE
                final_prediction = SAFE_FALLBACK_RESPONSE
        else:
            final_source = FinalSource.PRIMARY
            final_prediction = primary_prediction

        results.append(
            FallbackCaseResult(
                case_id=case_id,
                mode=mode,
                task=task,
                prompt=prompt,
                expected=expected,
                primary_prediction=primary_prediction,
                primary_action=primary_decision.action.value,
                primary_reason=primary_decision.reason,
                primary_violations=primary_violations,
                fallback_used=fallback_used,
                fallback_prediction=fallback_prediction,
                fallback_action=fallback_action,
                fallback_reason=fallback_reason,
                fallback_violations=fallback_violations,
                final_prediction=final_prediction,
                final_source=final_source,
                usage=usage,
                estimated_cost_usd=estimated_cost_usd,
            )
        )

    return _build_report(results)


def _build_report(results: list[FallbackCaseResult]) -> FallbackEvalReport:
    non_null_costs = [
        result.estimated_cost_usd
        for result in results
        if result.fallback_used and result.estimated_cost_usd is not None
    ]

    return FallbackEvalReport(
        cases=len(results),
        results=results,
        primary_action_counts=dict(
            Counter(result.primary_action for result in results)
        ),
        fallback_used_count=sum(1 for result in results if result.fallback_used),
        fallback_success_count=sum(
            1
            for result in results
            if result.final_source is FinalSource.FALLBACK
        ),
        final_source_counts=dict(
            Counter(result.final_source.value for result in results)
        ),
        unresolved_safe_count=sum(
            1 for result in results if result.final_source is FinalSource.SAFE
        ),
        primary_violation_counts=_count_violations(
            violation
            for result in results
            for violation in result.primary_violations
        ),
        final_violation_counts=_count_final_violations(results),
        estimated_total_cost_usd=sum(non_null_costs) if non_null_costs else None,
    )


def _count_final_violations(
    results: list[FallbackCaseResult],
) -> dict[str, int]:
    violations: list[dict[str, object]] = []
    for result in results:
        if result.final_source is FinalSource.PRIMARY:
            violations.extend(result.primary_violations)
        elif result.final_source in {FinalSource.FALLBACK, FinalSource.SAFE}:
            violations.extend(result.fallback_violations)

    return _count_violations(violations)


def _count_violations(
    violations: list[dict[str, object]] | object,
) -> dict[str, int]:
    return dict(
        Counter(
            violation["check"]
            for violation in violations
            if isinstance(violation, dict) and isinstance(violation.get("check"), str)
        )
    )


def _violation_dicts(violations: object) -> list[dict[str, object]]:
    return [
        violation.to_dict()
        for violation in violations
        if hasattr(violation, "to_dict")
    ]


def _require_str(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise FallbackEvalError(f"fallback eval record {field!r} must be a non-empty string")
    return value.strip()


def _require_prediction(record: dict[str, object]) -> str:
    value = record.get("prediction")
    if not isinstance(value, str):
        raise FallbackEvalError("fallback eval record 'prediction' must be a string")
    return value


def _require_expected(record: dict[str, object]) -> str | None:
    value = record.get("expected")
    if value is None or isinstance(value, str):
        return value
    raise FallbackEvalError("fallback eval record 'expected' must be a string or None")


def _require_int(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FallbackEvalError(f"fallback eval record {field!r} must be an integer")
    return value


def render_fallback_eval_markdown(report: FallbackEvalReport) -> str:
    lines = [
        "# OpenAI Fallback Eval Report",
        "",
        f"Cases: {report.cases}",
        "",
        "## Summary",
        "",
        f"- fallback_used_count: {report.fallback_used_count}",
        f"- fallback_success_count: {report.fallback_success_count}",
        f"- unresolved_safe_count: {report.unresolved_safe_count}",
        f"- estimated_total_cost_usd: {report.estimated_total_cost_usd}",
        "",
        "## Primary Action Counts",
        "",
    ]
    lines.extend(_counter_lines(report.primary_action_counts))
    lines.extend(["", "## Final Source Counts", ""])
    lines.extend(_counter_lines(report.final_source_counts))
    lines.extend(["", "## Primary Violation Counts", ""])
    lines.extend(_counter_lines(report.primary_violation_counts))
    lines.extend(["", "## Final Violation Counts", ""])
    lines.extend(_counter_lines(report.final_violation_counts))
    lines.extend(["", "## Fallback Cases", ""])

    fallback_results = [result for result in report.results if result.fallback_used]
    if not fallback_results:
        lines.append("- No fallback cases")

    for result in fallback_results:
        lines.extend(
            [
                f"### Case {result.case_id}: {result.mode} / {result.task}",
                "",
                f"- Primary action: {result.primary_action}",
                f"- Fallback action: {result.fallback_action}",
                f"- Final source: {result.final_source.value}",
                f"- Estimated cost USD: {result.estimated_cost_usd}",
            ]
        )
        if result.primary_violations:
            lines.append("- Primary violations:")
            for violation in result.primary_violations:
                lines.append(f"  - {violation['check']}: `{violation['term']}`")
        else:
            lines.append("- Primary violations: none")
        if result.fallback_violations:
            lines.append("- Fallback violations:")
            for violation in result.fallback_violations:
                lines.append(f"  - {violation['check']}: `{violation['term']}`")
        else:
            lines.append("- Fallback violations: none")
        lines.append("")

    return "\n".join(lines)


def write_fallback_eval_outputs(
    report: FallbackEvalReport,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "fallback_eval.json"
    markdown_path = target / "fallback_eval.md"
    predictions_path = target / "fallback_predictions.jsonl"

    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(render_fallback_eval_markdown(report), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in report.results:
            handle.write(
                json.dumps(_fallback_prediction_row(result), ensure_ascii=False) + "\n"
            )

    return json_path, markdown_path, predictions_path


def _fallback_prediction_row(result: FallbackCaseResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "mode": result.mode,
        "task": result.task,
        "prompt": result.prompt,
        "expected": result.expected,
        "prediction": result.final_prediction,
        "primary_prediction": result.primary_prediction,
        "primary_action": result.primary_action,
        "primary_reason": result.primary_reason,
        "fallback_used": result.fallback_used,
        "fallback_prediction": result.fallback_prediction,
        "fallback_action": result.fallback_action,
        "fallback_reason": result.fallback_reason,
        "final_source": result.final_source.value,
        "estimated_cost_usd": result.estimated_cost_usd,
    }


def _counter_lines(counter: dict[str, int]) -> list[str]:
    if not counter:
        return ["- none: 0"]
    return [f"- {key}: {value}" for key, value in counter.items()]
