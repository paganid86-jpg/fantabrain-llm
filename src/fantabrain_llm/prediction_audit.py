from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


class PredictionAuditError(ValueError):
    """Raised when prediction audit inputs are invalid."""


MODULE_PATTERN = re.compile(r"(?<!\d)\d-\d-\d(?:-\d)?(?!-?\d)")
MANTRA_FORBIDDEN_TERMS = ("modificatore", "modificatori", "reparto")
CLASSIC_MODULE_LANGUAGE = ("slot", "incastri", "moduli", "modulo principale")
MANTRA_ROLE_CODES = ("Pc", "T", "W", "A", "M", "E", "Dc", "Dd", "Ds")
AMBIGUOUS_SINGLE_LETTER_CODES = {"A", "E"}
ROLE_CODE_CONTEXT_PATTERN = r"(?:ruolo|codice|come|da|di|una|un|la|lo|per)"
MALFORMED_TERMS = (
    "offENSIVO",
    "multiruomo",
    "multiruoco",
    "sicurata",
    "attaccantini",
    "attorcicati",
    "inattaccante",
    "punteggia",
    "punteggiere",
    "migliro",
    "voti esattissimi",
)


@dataclass(frozen=True)
class AuditViolation:
    case_id: int
    mode: str
    task: str
    check: str
    term: str
    message: str
    hard_gate: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AuditReport:
    cases: int
    violations: list[AuditViolation]
    summary: dict[str, int]

    @property
    def hard_violation_count(self) -> int:
        return sum(1 for violation in self.violations if violation.hard_gate)

    def to_dict(self) -> dict[str, object]:
        return {
            "cases": self.cases,
            "summary": self.summary,
            "hard_violation_count": self.hard_violation_count,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def extract_modules(text: str) -> set[str]:
    return set(MODULE_PATTERN.findall(text))


def load_prediction_records(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if not source.exists():
        raise PredictionAuditError(f"Predictions file not found: {source}")

    records: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PredictionAuditError(
                    f"{source}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise PredictionAuditError(f"{source}:{line_number}: row must be an object")
            records.append(payload)

    if not records:
        raise PredictionAuditError(f"Predictions file has no rows: {source}")
    return records


def audit_prediction_records(records: Iterable[dict[str, object]]) -> AuditReport:
    record_list = list(records)
    violations: list[AuditViolation] = []

    for record in record_list:
        case_id = _require_int(record, "case_id")
        mode = _require_str(record, "mode")
        task = _require_str(record, "task")
        prompt = _require_str(record, "prompt")
        prediction = _require_str(record, "prediction")

        if mode == "mantra":
            violations.extend(_audit_mantra(case_id, mode, task, prompt, prediction))
        elif mode == "classic":
            violations.extend(_audit_classic(case_id, mode, task, prompt, prediction))
        else:
            raise PredictionAuditError(f"case {case_id}: unsupported mode {mode!r}")

        violations.extend(_audit_malformed_terms(case_id, mode, task, prediction))

    summary = dict(Counter(violation.check for violation in violations))
    return AuditReport(cases=len(record_list), violations=violations, summary=summary)


def render_audit_markdown(report: AuditReport) -> str:
    lines = [
        "# Prediction Audit",
        "",
        f"Cases: {report.cases}",
        f"Hard violations: {report.hard_violation_count}",
        "",
        "## Summary",
        "",
    ]
    if report.summary:
        for check, count in report.summary.items():
            lines.append(f"- {check}: {count}")
    else:
        lines.append("- No violations")

    lines.extend(["", "## Violations", ""])
    if not report.violations:
        lines.append("- No violations")
    for violation in report.violations:
        gate = "hard" if violation.hard_gate else "soft"
        lines.extend(
            [
                f"### Case {violation.case_id}: {violation.mode} / {violation.task}",
                "",
                f"- Check: {violation.check}",
                f"- Term: `{violation.term}`",
                f"- Gate: {gate}",
                f"- Message: {violation.message}",
                "",
            ]
        )

    return "\n".join(lines)


def write_audit_outputs(report: AuditReport, output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "prediction_audit.json"
    markdown_path = target / "prediction_audit.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(render_audit_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _audit_mantra(
    case_id: int,
    mode: str,
    task: str,
    prompt: str,
    prediction: str,
) -> list[AuditViolation]:
    violations: list[AuditViolation] = []
    lowered_prediction = prediction.lower()

    for term in MANTRA_FORBIDDEN_TERMS:
        if _contains_word(lowered_prediction, term):
            violations.append(
                AuditViolation(
                    case_id=case_id,
                    mode=mode,
                    task=task,
                    check="mantra_forbidden_terms",
                    term=term,
                    message=f"Mantra prediction uses forbidden term: {term}",
                    hard_gate=True,
                )
            )

    prompt_modules = extract_modules(prompt)
    prediction_modules = extract_modules(prediction)
    if prompt_modules:
        invented_modules = sorted(prediction_modules - prompt_modules)
        for module in invented_modules:
            violations.append(
                AuditViolation(
                    case_id=case_id,
                    mode=mode,
                    task=task,
                    check="invented_modules",
                    term=module,
                    message=f"Prediction mentions module not present in prompt: {module}",
                    hard_gate=True,
                )
            )
    else:
        for module in sorted(prediction_modules):
            violations.append(
                AuditViolation(
                    case_id=case_id,
                    mode=mode,
                    task=task,
                    check="invented_modules",
                    term=module,
                    message=(
                        "Prediction mentions a numeric module even though "
                        f"the prompt had none: {module}"
                    ),
                    hard_gate=True,
                )
            )

    return violations


def _audit_classic(
    case_id: int,
    mode: str,
    task: str,
    prompt: str,
    prediction: str,
) -> list[AuditViolation]:
    violations: list[AuditViolation] = []
    lowered_prediction = prediction.lower()

    for term in CLASSIC_MODULE_LANGUAGE:
        if _contains_word(lowered_prediction, term):
            violations.append(
                AuditViolation(
                    case_id=case_id,
                    mode=mode,
                    task=task,
                    check="classic_module_language",
                    term=term,
                    message=f"Classic prediction uses Mantra-style module language: {term}",
                    hard_gate=False,
                )
            )

    prompt_codes = _extract_role_codes(prompt)
    prediction_codes = _extract_role_codes(prediction)
    for code in MANTRA_ROLE_CODES:
        if code in prediction_codes and code not in prompt_codes:
            violations.append(
                AuditViolation(
                    case_id=case_id,
                    mode=mode,
                    task=task,
                    check="classic_role_code_leakage",
                    term=code,
                    message=f"Classic prediction leaks standalone Mantra role code: {code}",
                    hard_gate=True,
                )
            )

    return violations


def _audit_malformed_terms(
    case_id: int,
    mode: str,
    task: str,
    prediction: str,
) -> list[AuditViolation]:
    violations: list[AuditViolation] = []

    for term in MALFORMED_TERMS:
        if _contains_malformed_term(prediction, term):
            violations.append(
                AuditViolation(
                    case_id=case_id,
                    mode=mode,
                    task=task,
                    check="malformed_terms",
                    term=term,
                    message=f"Prediction contains known malformed term: {term}",
                    hard_gate=False,
                )
            )

    return violations


def _contains_malformed_term(text: str, term: str) -> bool:
    flags = 0 if any(character.isupper() for character in term) else re.IGNORECASE
    pattern = rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])"
    return re.search(pattern, text, flags=flags) is not None


def _extract_role_codes(text: str) -> set[str]:
    codes: set[str] = set()
    for code in MANTRA_ROLE_CODES:
        if code in AMBIGUOUS_SINGLE_LETTER_CODES:
            pattern = rf"(?<![A-Za-z]){ROLE_CODE_CONTEXT_PATTERN}\s+{code}(?![A-Za-z])"
        else:
            pattern = rf"(?<![A-Za-z]){re.escape(code)}(?![A-Za-z])"
        if re.search(pattern, text):
            codes.add(code)
    return codes


def _contains_word(lowered_text: str, lowered_term: str) -> bool:
    return (
        re.search(rf"(?<![A-Za-z]){re.escape(lowered_term)}(?![A-Za-z])", lowered_text)
        is not None
    )


def _require_str(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PredictionAuditError(f"prediction record {field!r} must be a non-empty string")
    return value.strip()


def _require_int(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PredictionAuditError(f"prediction record {field!r} must be an integer")
    return value
