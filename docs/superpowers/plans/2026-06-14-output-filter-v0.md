# Output Filter v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable output filter that classifies FantaBrain model answers as `pass`, `pass_with_warnings`, `fallback`, or `safe`.

**Architecture:** Reuse `src/fantabrain_llm/prediction_audit.py` for deterministic violation detection, then add a thin routing layer in `src/fantabrain_llm/output_filter.py`. Add a CLI that runs the filter over existing `predictions.jsonl` reports and writes JSON/Markdown summaries.

**Tech Stack:** Python stdlib, dataclasses, enums, JSONL, pytest, existing FantaBrain prediction audit utilities.

---

## File Structure

- Create `src/fantabrain_llm/output_filter.py`: single-answer filter API, decision types, report rendering/writing helpers.
- Create `scripts/filter_predictions.py`: CLI for applying the output filter to prediction runs.
- Create `tests/test_output_filter.py`: unit tests for single-answer routing and report rendering.
- Create `tests/test_filter_predictions_cli.py`: CLI tests for report writing and filesystem errors.
- Modify `README.md`: add the output filter command.

## Task 1: Core Filter API

**Files:**
- Create: `src/fantabrain_llm/output_filter.py`
- Test: `tests/test_output_filter.py`

- [ ] **Step 1: Write failing tests for the core filter API**

Create `tests/test_output_filter.py` with:

```python
from __future__ import annotations

from fantabrain_llm.output_filter import FilterAction, filter_model_output


def test_clean_mantra_output_passes() -> None:
    decision = filter_model_output(
        mode="mantra",
        task="lineup_advice",
        prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
        prediction="Sceglierei 3-4-2-1 se hai copertura sugli esterni e due T sicure.",
    )

    assert decision.action is FilterAction.PASS
    assert decision.reason == "no_violations"
    assert decision.violations == []


def test_malformed_only_output_passes_with_warnings() -> None:
    decision = filter_model_output(
        mode="classic",
        task="lineup_advice",
        prompt="Modalita Classic. Chi schiero?",
        prediction="Sceglierei il titolare, ma evita due punteggiere inutili.",
    )

    assert decision.action is FilterAction.PASS_WITH_WARNINGS
    assert decision.reason == "soft_violations"
    assert [violation.check for violation in decision.violations] == ["malformed_terms"]


def test_mantra_forbidden_term_triggers_fallback() -> None:
    decision = filter_model_output(
        mode="mantra",
        task="risk_management",
        prompt="Modalita Mantra. Ho pochi esterni: che faccio?",
        prediction="Usa il modificatore per proteggere il reparto.",
    )

    assert decision.action is FilterAction.FALLBACK
    assert decision.reason == "hard_violations"
    assert {violation.check for violation in decision.violations} == {"mantra_forbidden_terms"}


def test_mantra_invented_module_triggers_fallback() -> None:
    decision = filter_model_output(
        mode="mantra",
        task="lineup_advice",
        prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
        prediction="Sceglierei 4-5-1 per coprirti meglio.",
    )

    assert decision.action is FilterAction.FALLBACK
    assert decision.reason == "hard_violations"
    assert [violation.check for violation in decision.violations] == ["invented_modules"]


def test_classic_role_code_leakage_triggers_fallback() -> None:
    decision = filter_model_output(
        mode="classic",
        task="lineup_advice",
        prompt="Modalita Classic. Chi metto in attacco?",
        prediction="La T e una scelta forte se vuoi piu bonus.",
    )

    assert decision.action is FilterAction.FALLBACK
    assert decision.reason == "hard_violations"
    assert [violation.check for violation in decision.violations] == ["classic_role_code_leakage"]


def test_empty_output_returns_safe() -> None:
    decision = filter_model_output(
        mode="classic",
        task="auction_advice",
        prompt="Modalita Classic. Ultimo slot: che faccio?",
        prediction="   ",
    )

    assert decision.action is FilterAction.SAFE
    assert decision.reason == "empty_prediction"
    assert decision.violations == []


def test_hard_violation_after_fallback_returns_safe() -> None:
    decision = filter_model_output(
        mode="mantra",
        task="lineup_advice",
        prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
        prediction="Sceglierei 4-5-1.",
        fallback_failed=True,
    )

    assert decision.action is FilterAction.SAFE
    assert decision.reason == "fallback_failed_hard_violations"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_output_filter.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'fantabrain_llm.output_filter'
```

- [ ] **Step 3: Implement the minimal core API**

Create `src/fantabrain_llm/output_filter.py` with:

```python
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
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
        raise OutputFilterError(f"unsupported output filter preset: {preset}")
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
    violations = report.violations
    hard_count = sum(1 for violation in violations if violation.hard_gate)
    if hard_count and fallback_failed:
        return FilterDecision(
            action=FilterAction.SAFE,
            reason="fallback_failed_hard_violations",
            violations=violations,
        )
    if hard_count:
        return FilterDecision(
            action=FilterAction.FALLBACK,
            reason="hard_violations",
            violations=violations,
        )
    if violations:
        return FilterDecision(
            action=FilterAction.PASS_WITH_WARNINGS,
            reason="soft_violations",
            violations=violations,
        )
    return FilterDecision(
        action=FilterAction.PASS,
        reason="no_violations",
        violations=[],
    )
```

- [ ] **Step 4: Run core tests to verify they pass**

Run:

```bash
python -m pytest tests/test_output_filter.py -q
```

Expected:

```text
7 passed
```

- [ ] **Step 5: Commit core filter**

Run:

```bash
git add src/fantabrain_llm/output_filter.py tests/test_output_filter.py
git commit -m "feat: add output filter core"
```

## Task 2: Filter Run Reports

**Files:**
- Modify: `src/fantabrain_llm/output_filter.py`
- Modify: `tests/test_output_filter.py`

- [ ] **Step 1: Write failing tests for report helpers**

Append to `tests/test_output_filter.py`:

```python
import json
from pathlib import Path

from fantabrain_llm.output_filter import (
    filter_prediction_records,
    render_filter_markdown,
    write_filter_outputs,
)


def prediction_record(
    *,
    case_id: int,
    mode: str,
    prompt: str,
    prediction: str,
    task: str = "lineup_advice",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "mode": mode,
        "task": task,
        "prompt": prompt,
        "prediction": prediction,
        "expected": "Gold",
        "provider": "test",
        "model": "test",
    }


def test_filter_prediction_records_counts_actions() -> None:
    report = filter_prediction_records(
        [
            prediction_record(
                case_id=1,
                mode="mantra",
                prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
                prediction="Sceglierei 3-4-2-1.",
            ),
            prediction_record(
                case_id=2,
                mode="mantra",
                prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
                prediction="Sceglierei 4-5-1.",
            ),
        ]
    )

    assert report.cases == 2
    assert report.decision_counts == {"pass": 1, "fallback": 1}
    assert report.violation_counts == {"invented_modules": 1}
    assert report.results[1].decision.action is FilterAction.FALLBACK


def test_render_filter_markdown_includes_summary_and_cases() -> None:
    report = filter_prediction_records(
        [
            prediction_record(
                case_id=2,
                mode="mantra",
                prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
                prediction="Sceglierei 4-5-1.",
            )
        ]
    )

    markdown = render_filter_markdown(report)

    assert "# Output Filter Report" in markdown
    assert "fallback: 1" in markdown
    assert "Case 2" in markdown
    assert "invented_modules" in markdown


def test_write_filter_outputs_writes_json_and_markdown(tmp_path: Path) -> None:
    report = filter_prediction_records(
        [
            prediction_record(
                case_id=1,
                mode="classic",
                prompt="Modalita Classic. Chi schiero?",
                prediction="Sceglierei il titolare.",
            )
        ]
    )

    json_path, markdown_path = write_filter_outputs(report, tmp_path)

    assert json_path.name == "output_filter.json"
    assert markdown_path.name == "output_filter.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["cases"] == 1
    assert payload["decision_counts"] == {"pass": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_output_filter.py -q
```

Expected:

```text
ImportError: cannot import name 'filter_prediction_records'
```

- [ ] **Step 3: Implement report helpers**

Extend `src/fantabrain_llm/output_filter.py` with:

```python
import json
from pathlib import Path
```

and:

```python
@dataclass(frozen=True)
class FilteredPrediction:
    case_id: int
    mode: str
    task: str
    decision: FilterDecision

    def to_dict(self) -> dict[str, Any]:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "decision_counts": self.decision_counts,
            "violation_counts": self.violation_counts,
            "results": [result.to_dict() for result in self.results],
        }


def filter_prediction_records(records: Iterable[dict[str, object]]) -> FilterReport:
    results: list[FilteredPrediction] = []
    for record in records:
        case_id = _require_int(record, "case_id")
        mode = _require_str(record, "mode")
        task = _require_str(record, "task")
        prompt = _require_str(record, "prompt")
        prediction = _require_str(record, "prediction")
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

    decision_counts = dict(Counter(result.decision.action.value for result in results))
    violation_counts = dict(
        Counter(
            violation.check
            for result in results
            for violation in result.decision.violations
        )
    )
    return FilterReport(
        cases=len(results),
        results=results,
        decision_counts=decision_counts,
        violation_counts=violation_counts,
    )


def render_filter_markdown(report: FilterReport) -> str:
    lines = [
        "# Output Filter Report",
        "",
        f"Cases: {report.cases}",
        "",
        "## Decisions",
        "",
    ]
    if report.decision_counts:
        for action, count in report.decision_counts.items():
            lines.append(f"- {action}: {count}")
    else:
        lines.append("- No cases")

    lines.extend(["", "## Violations", ""])
    if report.violation_counts:
        for check, count in report.violation_counts.items():
            lines.append(f"- {check}: {count}")
    else:
        lines.append("- No violations")

    flagged = [
        result
        for result in report.results
        if result.decision.action is not FilterAction.PASS
    ]
    lines.extend(["", "## Flagged Cases", ""])
    if not flagged:
        lines.append("- No flagged cases")
    for result in flagged:
        lines.extend(
            [
                f"### Case {result.case_id}: {result.mode} / {result.task}",
                "",
                f"- Action: `{result.decision.action.value}`",
                f"- Reason: `{result.decision.reason}`",
            ]
        )
        for violation in result.decision.violations:
            gate = "hard" if violation.hard_gate else "soft"
            lines.append(
                f"- {violation.check} ({gate}): `{violation.term}` - {violation.message}"
            )
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
    if not isinstance(value, str):
        raise OutputFilterError(f"prediction record {field!r} must be a string")
    return value


def _require_int(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutputFilterError(f"prediction record {field!r} must be an integer")
    return value
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_output_filter.py -q
```

Expected:

```text
10 passed
```

- [ ] **Step 5: Commit report helpers**

Run:

```bash
git add src/fantabrain_llm/output_filter.py tests/test_output_filter.py
git commit -m "feat: add output filter reports"
```

## Task 3: CLI for Prediction Runs

**Files:**
- Create: `scripts/filter_predictions.py`
- Test: `tests/test_filter_predictions_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_filter_predictions_cli.py` with:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def prediction(case_id: int, prediction_text: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "mode": "mantra",
        "task": "lineup_advice",
        "tags": ["test"],
        "prompt": "Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
        "expected": "Gold",
        "prediction": prediction_text,
        "provider": "test",
        "model": "test",
    }


def test_filter_predictions_cli_writes_outputs(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps(prediction(1, "Sceglierei 3-4-2-1.")),
                json.dumps(prediction(2, "Sceglierei 4-5-1.")),
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "filter"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/filter_predictions.py",
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Output filter JSON written to" in result.stdout
    assert "fallback: 1" in result.stdout
    assert (output_dir / "output_filter.json").exists()
    assert (output_dir / "output_filter.md").exists()


def test_filter_predictions_cli_reports_errors_cleanly(tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("already here", encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(prediction(1, "Sceglierei 3-4-2-1.")),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/filter_predictions.py",
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_file),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Output filter error:" in result.stderr
    assert "Traceback" not in result.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_filter_predictions_cli.py -q
```

Expected:

```text
can't open file 'scripts/filter_predictions.py'
```

- [ ] **Step 3: Implement CLI**

Create `scripts/filter_predictions.py` with:

```python
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
    parser = argparse.ArgumentParser(
        description="Apply the FantaBrain output filter to a prediction run."
    )
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
    for action, count in report.decision_counts.items():
        print(f"{action}: {count}")
    for check, count in report.violation_counts.items():
        print(f"{check}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests to verify they pass**

Run:

```bash
python -m pytest tests/test_filter_predictions_cli.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit CLI**

Run:

```bash
git add scripts/filter_predictions.py tests/test_filter_predictions_cli.py
git commit -m "feat: add output filter cli"
```

## Task 4: README and Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README command section**

Add this command near the existing prediction/audit commands:

Apply the app-oriented output filter to a prediction run:

```bash
python scripts/filter_predictions.py \
  --predictions reports/runs/<run-name>/predictions.jsonl \
  --output-dir reports/runs/<run-name>
```

- [ ] **Step 2: Run targeted tests**

Run:

```bash
python -m pytest tests/test_output_filter.py tests/test_filter_predictions_cli.py -q
```

Expected:

```text
12 passed
```

- [ ] **Step 3: Run full tests**

Run:

```bash
python -m pytest -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 4: Run lint on touched files**

Run:

```bash
python -m ruff check src/fantabrain_llm/output_filter.py scripts/filter_predictions.py tests/test_output_filter.py tests/test_filter_predictions_cli.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 5: Commit docs**

Run:

```bash
git add README.md
git commit -m "docs: document output filter command"
```

## Task 5: Push and Memory Update

**Files:**
- Modify external memory after code is complete: `C:\Users\DantePagani\llm-memory\wiki\projects\fantabrain-llm\project-overview.md`

- [ ] **Step 1: Inspect final status**

Run:

```bash
git status -sb
git log --oneline -5
```

Expected:

```text
## codex/output-filter-v0...origin/master [ahead N]
```

- [ ] **Step 2: Push branch**

Run:

```bash
git push -u origin codex/output-filter-v0
```

Expected:

```text
branch 'codex/output-filter-v0' set up to track 'origin/codex/output-filter-v0'
```

- [ ] **Step 3: Update llm-memory**

Append a short checkpoint noting:

- branch `codex/output-filter-v0`;
- core filter API;
- CLI report files `output_filter.json` and `output_filter.md`;
- verification results;
- next step: PR/open review, then app integration/fallback branch.

- [ ] **Step 4: Final response**

Report:

- branch name;
- commits created;
- verification commands and results;
- PR URL or push status;
- memory update status.
