from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fantabrain_llm.prediction_audit import AuditViolation, audit_prediction_records


class OutputFilterError(ValueError):
    """Raised when output filter inputs are invalid."""


class FilterAction(str, Enum):
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
