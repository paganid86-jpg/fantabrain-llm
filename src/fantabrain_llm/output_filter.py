from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from fantabrain_llm.prediction_audit import AuditViolation, audit_prediction_records


class OutputFilterError(ValueError):
    """Raised when output filter inputs are invalid."""


class FilterAction(StrEnum):
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    FALLBACK = "fallback"
    SAFE = "safe"


@dataclass(frozen=True)
class FilterDecision:
    action: FilterAction
    reason: str
    violations: list[AuditViolation]

    @property
    def hard_violation_count(self) -> int:
        return sum(1 for violation in self.violations if violation.hard_gate)

    @property
    def soft_violation_count(self) -> int:
        return sum(1 for violation in self.violations if not violation.hard_gate)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "hard_violation_count": self.hard_violation_count,
            "soft_violation_count": self.soft_violation_count,
            "violations": [violation.to_dict() for violation in self.violations],
        }


@dataclass(frozen=True)
class FilteredPrediction:
    case_id: int
    mode: str
    task: str
    decision: FilterDecision

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "task": self.task,
            "decision": self.decision.to_dict(),
        }


@dataclass(frozen=True)
class FilterReport:
    cases: int
    results: list[FilteredPrediction]
    decision_counts: dict[str, int]
    violation_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "cases": self.cases,
            "decision_counts": self.decision_counts,
            "violation_counts": self.violation_counts,
            "results": [result.to_dict() for result in self.results],
        }


def filter_model_output(
    *,
    mode: str,
    task: str,
    prompt: str,
    prediction: str,
    preset: str = "app_v0",
    fallback_failed: bool = False,
) -> FilterDecision:
    if preset != "app_v0":
        raise OutputFilterError(f"Unsupported output filter preset: {preset}")

    if not prediction.strip():
        return FilterDecision(
            action=FilterAction.SAFE,
            reason="empty_prediction",
            violations=[],
        )

    report = audit_prediction_records(
        [
            {
                "case_id": 1,
                "mode": mode,
                "task": task,
                "prompt": prompt,
                "prediction": prediction,
            }
        ]
    )
    decision = FilterDecision(
        action=FilterAction.PASS,
        reason="no_violations",
        violations=report.violations,
    )

    if decision.hard_violation_count and fallback_failed:
        return FilterDecision(
            action=FilterAction.SAFE,
            reason="fallback_failed_hard_violations",
            violations=report.violations,
        )
    if decision.hard_violation_count:
        return FilterDecision(
            action=FilterAction.FALLBACK,
            reason="hard_violations",
            violations=report.violations,
        )
    if decision.soft_violation_count:
        return FilterDecision(
            action=FilterAction.PASS_WITH_WARNINGS,
            reason="soft_violations",
            violations=report.violations,
        )
    return decision


def filter_prediction_records(records: Iterable[dict[str, object]]) -> FilterReport:
    results: list[FilteredPrediction] = []

    for record in records:
        case_id = _require_int(record, "case_id")
        mode = _require_str(record, "mode")
        task = _require_str(record, "task")
        prompt = _require_str(record, "prompt")
        prediction = _require_prediction(record)

        decision = filter_model_output(
            mode=mode,
            task=task,
            prompt=prompt,
            prediction=prediction,
        )
        results.append(
            FilteredPrediction(
                case_id=case_id,
                mode=mode,
                task=task,
                decision=decision,
            )
        )

    decision_counts = Counter(result.decision.action.value for result in results)
    violation_counts = Counter(
        violation.check
        for result in results
        for violation in result.decision.violations
    )
    return FilterReport(
        cases=len(results),
        results=results,
        decision_counts=dict(decision_counts),
        violation_counts=dict(violation_counts),
    )


def render_filter_markdown(report: FilterReport) -> str:
    lines = [
        "# Output Filter Report",
        "",
        f"Cases: {report.cases}",
        "",
        "## Decision Counts",
        "",
    ]
    if report.decision_counts:
        for action, count in report.decision_counts.items():
            lines.append(f"- {action}: {count}")
    else:
        lines.append("- No decisions")

    lines.extend(["", "## Violation Counts", ""])
    if report.violation_counts:
        for check, count in report.violation_counts.items():
            lines.append(f"- {check}: {count}")
    else:
        lines.append("- No violations")

    lines.extend(["", "## Flagged Cases", ""])
    flagged_results = [
        result
        for result in report.results
        if result.decision.action is not FilterAction.PASS
    ]
    if not flagged_results:
        lines.append("- No flagged cases")

    for result in flagged_results:
        lines.extend(
            [
                f"### Case {result.case_id}: {result.mode} / {result.task}",
                "",
                f"- Action: {result.decision.action.value}",
                f"- Reason: {result.decision.reason}",
            ]
        )
        if result.decision.violations:
            lines.append("- Violations:")
            for violation in result.decision.violations:
                gate = "hard" if violation.hard_gate else "soft"
                lines.append(f"  - {violation.check}: `{violation.term}` ({gate})")
        else:
            lines.append("- Violations: none")
        lines.append("")

    return "\n".join(lines)


def write_filter_outputs(report: FilterReport, output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "output_filter.json"
    markdown_path = target / "output_filter.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(render_filter_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _require_str(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise OutputFilterError(f"prediction record {field!r} must be a non-empty string")
    return value.strip()


def _require_prediction(record: dict[str, object]) -> str:
    value = record.get("prediction")
    if not isinstance(value, str):
        raise OutputFilterError("prediction record 'prediction' must be a string")
    return value


def _require_int(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutputFilterError(f"prediction record {field!r} must be an integer")
    return value
