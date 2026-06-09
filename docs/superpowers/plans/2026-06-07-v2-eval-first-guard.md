# V2 Eval-First Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repeatable eval-first path that runs the v2 adapter with a stricter mode-aware prompt guard and audits generated predictions for known v4/v5 failure patterns.

**Architecture:** Keep the guard as a pure message transformation before inference, then keep prediction auditing as a deterministic text pass over `predictions.jsonl`. This keeps training data, Pagella v0, and model outputs untouched while giving us measurable evidence before any new dataset work.

**Tech Stack:** Python standard library, existing `fantabrain_llm` schema/dataset/predictions modules, pytest, current CLI scripts.

---

## File Structure

- Create `src/fantabrain_llm/prompt_guards.py`: prompt guard presets and pure `apply_prompt_guard()` helper.
- Create `src/fantabrain_llm/prediction_audit.py`: deterministic prediction audit checks, report dataclasses, JSON/Markdown rendering.
- Modify `scripts/generate_predictions.py`: add `--prompt-guard` and apply it before `client.generate()`.
- Create `scripts/audit_predictions.py`: CLI wrapper for `prediction_audit`.
- Create `tests/test_prompt_guards.py`: unit tests for guard injection.
- Create `tests/test_prediction_audit.py`: unit tests for invented modules, forbidden terms, token-aware role code checks, and reports.
- Modify `tests/test_generate_predictions_cli.py`: echo-provider smoke test for `--prompt-guard`.
- Create `docs/runbooks/qwen25-v2-eval-first-guard.md`: Colab/runbook commands for the guarded v2 eval.
- Modify `README.md`: add the guarded v2 eval commands near the existing Pagella adapter commands.

---

### Task 1: Prompt Guard Module

**Files:**
- Create: `src/fantabrain_llm/prompt_guards.py`
- Test: `tests/test_prompt_guards.py`

- [ ] **Step 1: Write failing prompt guard tests**

Create `tests/test_prompt_guards.py`:

```python
from __future__ import annotations

import pytest

from fantabrain_llm.prompt_guards import (
    PromptGuardError,
    apply_prompt_guard,
    prompt_guard_names,
)
from fantabrain_llm.schema import ChatMessage


def messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="Sei il coach AI privato di FantaBrain."),
        ChatMessage(role="user", content="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?"),
    ]


def test_prompt_guard_names_exposes_none_and_mode_fence() -> None:
    assert prompt_guard_names() == ["none", "mode_fence_v1"]


def test_none_prompt_guard_returns_copy_without_changing_content() -> None:
    guarded = apply_prompt_guard(messages(), mode="mantra", preset="none")

    assert guarded == messages()
    assert guarded is not messages()


def test_mode_fence_merges_guard_into_single_system_message_for_mantra() -> None:
    guarded = apply_prompt_guard(messages(), mode="mantra", preset="mode_fence_v1")

    assert [message.role for message in guarded] == ["system", "user"]
    assert guarded[0].content.count("Sei il coach AI privato di FantaBrain.") == 1
    assert "Prompt guard mode_fence_v1" in guarded[0].content
    assert "Regole Mantra" in guarded[0].content
    assert "Non usare modificatore" in guarded[0].content
    assert "Regole Classic" not in guarded[0].content
    assert guarded[1].content == "Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?"


def test_mode_fence_adds_classic_rules_for_classic() -> None:
    guarded = apply_prompt_guard(messages(), mode="classic", preset="mode_fence_v1")

    assert "Regole Classic" in guarded[0].content
    assert "Non usare codici ruolo Mantra" in guarded[0].content
    assert "Regole Mantra" not in guarded[0].content


def test_prompt_guard_rejects_unknown_preset() -> None:
    with pytest.raises(PromptGuardError, match="Unknown prompt guard"):
        apply_prompt_guard(messages(), mode="mantra", preset="missing")


def test_prompt_guard_requires_initial_system_message() -> None:
    with pytest.raises(PromptGuardError, match="first message must be system"):
        apply_prompt_guard(
            [ChatMessage(role="user", content="Domanda")],
            mode="mantra",
            preset="mode_fence_v1",
        )
```

- [ ] **Step 2: Run prompt guard tests to verify they fail**

Run:

```bash
pytest tests/test_prompt_guards.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'fantabrain_llm.prompt_guards'`.

- [ ] **Step 3: Implement `prompt_guards.py`**

Create `src/fantabrain_llm/prompt_guards.py`:

```python
from __future__ import annotations

from fantabrain_llm.schema import ChatMessage


class PromptGuardError(ValueError):
    """Raised when a prompt guard cannot be applied."""


SHARED_MODE_FENCE_RULES = """Regole condivise:
- Rispondi in italiano pulito e naturale.
- Inizia con la decisione o con il rifiuto motivato.
- Non inventare nomi di giocatori, fatti live, voti futuri esatti o probabilita non disponibili.
- Se mancano dati chiave, di cosa manca e dai solo un criterio generale.
- Evita parole inventate o malformate."""

MANTRA_MODE_FENCE_RULES = """Regole Mantra:
- Ragiona con codici ruolo, copertura slot, vincoli di modulo e compatibilita della panchina.
- Non usare modificatore, modificatori o reparto.
- Se il prompt cita moduli specifici, non introdurre moduli extra.
- Se il prompt non cita moduli numerici, non inventare un numero di modulo."""

CLASSIC_MODE_FENCE_RULES = """Regole Classic:
- Ragiona con reparti, titolarita, bonus, malus, modificatore difesa quando rilevante e panchina per reparto.
- Non usare codici ruolo Mantra come Pc, T, W, A, M, E, Dc, Dd o Ds se l'utente non chiede Mantra.
- Non parlare di incastri di modulo come se Classic fosse Mantra."""


def prompt_guard_names() -> list[str]:
    return ["none", "mode_fence_v1"]


def apply_prompt_guard(
    messages: list[ChatMessage],
    *,
    mode: str,
    preset: str = "none",
) -> list[ChatMessage]:
    normalized = preset.strip().lower()
    if normalized == "none":
        return list(messages)
    if normalized != "mode_fence_v1":
        raise PromptGuardError(f"Unknown prompt guard: {preset}")
    if not messages or messages[0].role != "system":
        raise PromptGuardError("prompt guard requires the first message must be system")

    guard = _mode_fence_v1(mode)
    guarded_system = ChatMessage(
        role="system",
        content=f"{messages[0].content}\n\n{guard}",
    )
    return [guarded_system, *messages[1:]]


def _mode_fence_v1(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized == "mantra":
        mode_rules = MANTRA_MODE_FENCE_RULES
    elif normalized == "classic":
        mode_rules = CLASSIC_MODE_FENCE_RULES
    else:
        raise PromptGuardError(f"Unsupported mode for prompt guard: {mode}")

    return "\n\n".join(
        [
            "Prompt guard mode_fence_v1.",
            SHARED_MODE_FENCE_RULES,
            mode_rules,
        ]
    )
```

- [ ] **Step 4: Run prompt guard tests to verify they pass**

Run:

```bash
pytest tests/test_prompt_guards.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/fantabrain_llm/prompt_guards.py tests/test_prompt_guards.py
git commit -m "feat: add mode-aware prompt guard"
```

---

### Task 2: Generate Predictions Guard Wiring

**Files:**
- Modify: `scripts/generate_predictions.py`
- Modify: `tests/test_generate_predictions_cli.py`

- [ ] **Step 1: Add failing CLI smoke test**

Append this test to `tests/test_generate_predictions_cli.py`:

```python
def test_generate_predictions_cli_applies_prompt_guard_with_echo(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "mode": "mantra",
                        "task": "lineup_advice",
                        "source": "test",
                        "quality_score": 5,
                        "tags": ["test"],
                        "messages": [
                            {"role": "system", "content": "System"},
                            {"role": "user", "content": "Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?"},
                            {"role": "assistant", "content": "Gold"},
                        ],
                    }
                )
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_predictions.py",
            "--provider",
            "echo",
            "--model",
            "echo-baseline",
            "--eval",
            str(eval_path),
            "--run-name",
            "echo-guarded",
            "--output-root",
            str(output_root),
            "--prompt-guard",
            "mode_fence_v1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = output_root / "echo-guarded"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert "Prediction run written" in result.stdout
    assert summary["prompt_guard"] == "mode_fence_v1"
```

If the file currently lacks imports, ensure the top of `tests/test_generate_predictions_cli.py` includes:

```python
import json
import subprocess
import sys
from pathlib import Path
```

- [ ] **Step 2: Run CLI test to verify it fails**

Run:

```bash
pytest tests/test_generate_predictions_cli.py::test_generate_predictions_cli_applies_prompt_guard_with_echo -q
```

Expected: fail because `generate_predictions.py` does not accept `--prompt-guard`.

- [ ] **Step 3: Wire prompt guard into `generate_predictions.py`**

Modify `scripts/generate_predictions.py` imports:

```python
from fantabrain_llm.prompt_guards import (  # noqa: E402
    PromptGuardError,
    apply_prompt_guard,
    prompt_guard_names,
)
```

Add parser argument after `--torch-dtype`:

```python
    parser.add_argument(
        "--prompt-guard",
        default="none",
        choices=prompt_guard_names(),
        help="Optional inference-time prompt guard preset.",
    )
```

Replace the generation append block with:

```python
            prompt_messages = apply_prompt_guard(
                to_generation_messages(example),
                mode=example.mode,
                preset=args.prompt_guard,
            )
            responses.append(
                client.generate(
                    prompt_messages,
                    mode=example.mode,
                    task=example.task,
                )
            )
```

Add `PromptGuardError` to the caught exception tuple:

```python
    except (DatasetError, InferenceError, PromptGuardError) as exc:
```

Add guard metadata in `write_prediction_run()`:

```python
                "prompt_guard": args.prompt_guard,
```

- [ ] **Step 4: Run prompt guard and CLI tests**

Run:

```bash
pytest tests/test_prompt_guards.py tests/test_generate_predictions_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add scripts/generate_predictions.py tests/test_generate_predictions_cli.py
git commit -m "feat: apply prompt guards during prediction generation"
```

---

### Task 3: Prediction Audit Core

**Files:**
- Create: `src/fantabrain_llm/prediction_audit.py`
- Test: `tests/test_prediction_audit.py`

- [ ] **Step 1: Write failing audit tests**

Create `tests/test_prediction_audit.py`:

```python
from __future__ import annotations

from fantabrain_llm.prediction_audit import (
    audit_prediction_records,
    extract_modules,
    render_audit_markdown,
)


def prediction(
    *,
    case_id: int,
    mode: str,
    prompt: str,
    text: str,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "mode": mode,
        "task": "lineup_advice",
        "tags": ["test"],
        "prompt": prompt,
        "expected": "Gold",
        "prediction": text,
        "provider": "echo",
        "model": "echo",
    }


def test_extract_modules_uses_full_module_tokens() -> None:
    assert extract_modules("Meglio 3-4-2-1 o 4-3-3?") == {"3-4-2-1", "4-3-3"}
    assert "3-4-2" not in extract_modules("Meglio 3-4-2-1?")


def test_audit_flags_mantra_forbidden_terms_and_invented_modules() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=2,
                mode="mantra",
                prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
                text="Sceglierei 4-5-1. Il modificatore aiuta.",
            )
        ]
    )

    assert report.summary == {"invented_modules": 1, "mantra_forbidden_terms": 1}
    assert report.hard_violation_count == 2
    assert {violation.check for violation in report.violations} == {
        "invented_modules",
        "mantra_forbidden_terms",
    }


def test_audit_allows_classic_role_codes_present_in_prompt() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=1,
                mode="classic",
                prompt="Classic ma sto confrontando la vecchia T Mantra con un attaccante.",
                text="La T citata nel prompt non e leakage nuovo.",
            )
        ]
    )

    assert report.violations == []


def test_audit_flags_classic_role_code_leakage_token_aware() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=7,
                mode="classic",
                prompt="Modalita Classic. Chi metto capitano?",
                text="La T e una scelta forte, ma meglio valutare slot e moduli.",
            )
        ]
    )

    checks = [violation.check for violation in report.violations]
    assert checks == ["classic_module_language", "classic_role_code_leakage"]
    assert report.hard_violation_count == 1


def test_audit_flags_malformed_terms_as_soft_violations() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=35,
                mode="classic",
                prompt="Modalita Classic. Sono primo.",
                text="La formazione conserva evita due punteggiere.",
            )
        ]
    )

    assert report.summary == {"malformed_terms": 1}
    assert report.hard_violation_count == 0


def test_render_audit_markdown_includes_summary_and_case_details() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=3,
                mode="mantra",
                prompt="Modalita Mantra. Che faccio?",
                text="Usa il modificatore.",
            )
        ]
    )

    markdown = render_audit_markdown(report)

    assert "# Prediction Audit" in markdown
    assert "mantra_forbidden_terms: 1" in markdown
    assert "Case 3" in markdown
    assert "modificatore" in markdown
```

- [ ] **Step 2: Run audit tests to verify they fail**

Run:

```bash
pytest tests/test_prediction_audit.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'fantabrain_llm.prediction_audit'`.

- [ ] **Step 3: Implement `prediction_audit.py`**

Create `src/fantabrain_llm/prediction_audit.py`:

```python
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class PredictionAuditError(ValueError):
    """Raised when prediction audit inputs are invalid."""


MODULE_PATTERN = re.compile(r"(?<!\d)(?:\d-\d-\d(?:-\d)?)(?!-\d|\d)")
MANTRA_FORBIDDEN_TERMS = ("modificatore", "modificatori", "reparto")
CLASSIC_MODULE_LANGUAGE = ("slot", "incastri", "moduli", "modulo principale")
MANTRA_ROLE_CODES = ("Pc", "T", "W", "A", "M", "E", "Dc", "Dd", "Ds")
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
    check: str
    term: str
    hard: bool
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionAuditReport:
    cases: int
    summary: dict[str, int]
    violations: list[AuditViolation]

    @property
    def hard_violation_count(self) -> int:
        return sum(1 for violation in self.violations if violation.hard)

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


def audit_prediction_records(records: list[dict[str, object]]) -> PredictionAuditReport:
    violations: list[AuditViolation] = []
    for record in records:
        case_id = _require_int(record, "case_id")
        mode = _require_str(record, "mode")
        prompt = _require_str(record, "prompt")
        prediction = _require_str(record, "prediction")

        if mode == "mantra":
            violations.extend(_audit_mantra(case_id, prompt, prediction))
        elif mode == "classic":
            violations.extend(_audit_classic(case_id, prompt, prediction))
        else:
            raise PredictionAuditError(f"case {case_id}: unsupported mode {mode!r}")

        violations.extend(_audit_malformed_terms(case_id, mode, prediction))

    summary = dict(Counter(violation.check for violation in violations))
    return PredictionAuditReport(
        cases=len(records),
        summary=summary,
        violations=violations,
    )


def render_audit_markdown(report: PredictionAuditReport) -> str:
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
        for check, count in sorted(report.summary.items()):
            lines.append(f"- {check}: {count}")
    else:
        lines.append("- No violations")

    lines.extend(["", "## Violations", ""])
    if not report.violations:
        lines.append("No violations found.")
    for violation in report.violations:
        severity = "hard" if violation.hard else "soft"
        lines.extend(
            [
                f"### Case {violation.case_id}: {violation.check}",
                "",
                f"- Mode: {violation.mode}",
                f"- Severity: {severity}",
                f"- Term: `{violation.term}`",
                f"- Note: {violation.note}",
                "",
            ]
        )
    return "\n".join(lines)


def write_audit_outputs(
    report: PredictionAuditReport,
    output_dir: str | Path,
) -> tuple[Path, Path]:
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


def _audit_mantra(case_id: int, prompt: str, prediction: str) -> list[AuditViolation]:
    violations: list[AuditViolation] = []
    lowered = prediction.lower()
    for term in MANTRA_FORBIDDEN_TERMS:
        if _contains_lower_term(lowered, term):
            violations.append(
                AuditViolation(
                    case_id=case_id,
                    mode="mantra",
                    check="mantra_forbidden_terms",
                    term=term,
                    hard=True,
                    note="Mantra prediction contains a forbidden Classic term.",
                )
            )

    prompt_modules = extract_modules(prompt)
    prediction_modules = extract_modules(prediction)
    if prompt_modules:
        for module in sorted(prediction_modules - prompt_modules):
            violations.append(
                AuditViolation(
                    case_id=case_id,
                    mode="mantra",
                    check="invented_modules",
                    term=module,
                    hard=True,
                    note="Prediction introduced a module not present in the prompt.",
                )
            )
    elif prediction_modules:
        for module in sorted(prediction_modules):
            violations.append(
                AuditViolation(
                    case_id=case_id,
                    mode="mantra",
                    check="invented_modules",
                    term=module,
                    hard=True,
                    note="Prediction invented a numeric module when the prompt had none.",
                )
            )
    return violations


def _audit_classic(case_id: int, prompt: str, prediction: str) -> list[AuditViolation]:
    violations: list[AuditViolation] = []
    lowered = prediction.lower()
    for term in CLASSIC_MODULE_LANGUAGE:
        if _contains_lower_term(lowered, term):
            violations.append(
                AuditViolation(
                    case_id=case_id,
                    mode="classic",
                    check="classic_module_language",
                    term=term,
                    hard=False,
                    note="Classic prediction uses Mantra-like module or slot language.",
                )
            )

    prompt_codes = _role_codes_in_text(prompt)
    prediction_codes = _role_codes_in_text(prediction)
    for code in sorted(prediction_codes - prompt_codes):
        violations.append(
            AuditViolation(
                case_id=case_id,
                mode="classic",
                check="classic_role_code_leakage",
                term=code,
                hard=True,
                note="Classic prediction introduced a Mantra role code.",
            )
        )
    return violations


def _audit_malformed_terms(case_id: int, mode: str, prediction: str) -> list[AuditViolation]:
    violations: list[AuditViolation] = []
    for term in MALFORMED_TERMS:
        if term in prediction:
            violations.append(
                AuditViolation(
                    case_id=case_id,
                    mode=mode,
                    check="malformed_terms",
                    term=term,
                    hard=False,
                    note="Prediction contains a known malformed term from previous audits.",
                )
            )
    return violations


def _role_codes_in_text(text: str) -> set[str]:
    found: set[str] = set()
    for code in MANTRA_ROLE_CODES:
        pattern = rf"(?<![A-Za-zÀ-ÿ]){re.escape(code)}(?![A-Za-zÀ-ÿ])"
        if re.search(pattern, text):
            found.add(code)
    return found


def _contains_lower_term(lowered_text: str, lowered_term: str) -> bool:
    pattern = rf"(?<![A-Za-zÀ-ÿ]){re.escape(lowered_term)}(?![A-Za-zÀ-ÿ])"
    return re.search(pattern, lowered_text) is not None


def _require_str(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PredictionAuditError(f"prediction record missing text field: {key}")
    return value


def _require_int(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise PredictionAuditError(f"prediction record missing integer field: {key}")
    return value
```

- [ ] **Step 4: Run audit tests**

Run:

```bash
pytest tests/test_prediction_audit.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add src/fantabrain_llm/prediction_audit.py tests/test_prediction_audit.py
git commit -m "feat: add deterministic prediction audit"
```

---

### Task 4: Prediction Audit CLI

**Files:**
- Create: `scripts/audit_predictions.py`
- Test: `tests/test_prediction_audit.py`

- [ ] **Step 1: Add failing CLI test**

Append this test to `tests/test_prediction_audit.py`:

```python
def test_audit_predictions_cli_writes_outputs_and_fails_on_hard_gates(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps(
                    prediction(
                        case_id=2,
                        mode="mantra",
                        prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
                        text="Sceglierei 4-5-1 con modificatore.",
                    )
                )
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_predictions.py",
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
            "--fail-on-hard-gates",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Hard violations: 2" in result.stdout
    assert (output_dir / "prediction_audit.json").exists()
    assert (output_dir / "prediction_audit.md").exists()
```

Ensure `tests/test_prediction_audit.py` imports:

```python
import json
import subprocess
import sys
from pathlib import Path
```

- [ ] **Step 2: Run CLI test to verify it fails**

Run:

```bash
pytest tests/test_prediction_audit.py::test_audit_predictions_cli_writes_outputs_and_fails_on_hard_gates -q
```

Expected: fail because `scripts/audit_predictions.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `scripts/audit_predictions.py`:

```python
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
    for check, count in sorted(report.summary.items()):
        print(f"{check}: {count}")

    if args.fail_on_hard_gates and report.hard_violation_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run audit tests**

Run:

```bash
pytest tests/test_prediction_audit.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add scripts/audit_predictions.py tests/test_prediction_audit.py
git commit -m "feat: add prediction audit cli"
```

---

### Task 5: Runbook and README

**Files:**
- Create: `docs/runbooks/qwen25-v2-eval-first-guard.md`
- Modify: `README.md`

- [ ] **Step 1: Create runbook**

Create `docs/runbooks/qwen25-v2-eval-first-guard.md`:

````markdown
# Qwen v2 Eval-First Guard Runbook

Use this runbook to evaluate the current champion adapter, `qwen25-3b-fantabrain-sft-v2`, with the `mode_fence_v1` prompt guard before creating any new dataset.

## Colab Setup

```python
# Controlla GPU e token.
import os
import torch

print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("GH_TOKEN presente:", bool(os.environ.get("GH_TOKEN")))
print("HF_TOKEN presente:", bool(os.environ.get("HF_TOKEN")))
```

```python
# Clona o aggiorna la repo.
import os
import subprocess
from pathlib import Path

repo = Path("/content/fantabrain-llm")
token = os.environ["GH_TOKEN"]
url = f"https://{token}@github.com/paganid86-jpg/fantabrain-llm.git"

if repo.exists():
    subprocess.run(["git", "-C", str(repo), "fetch", "origin", "master"], check=True)
    subprocess.run(["git", "-C", str(repo), "switch", "master"], check=True)
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", url, str(repo)], check=True)

%cd /content/fantabrain-llm
```

```python
# Installa dipendenze eval/training.
%cd /content/fantabrain-llm
!python -m pip install -q -e ".[train]"
```

## Restore Adapter

Upload `qwen25-3b-fantabrain-sft-v2-adapter.zip` in Colab, then run:

```python
# Estrai adapter v2.
from pathlib import Path
import zipfile

%cd /content/fantabrain-llm

zip_path = Path("/content/qwen25-3b-fantabrain-sft-v2-adapter.zip")
assert zip_path.exists(), "Carica qwen25-3b-fantabrain-sft-v2-adapter.zip in /content"

with zipfile.ZipFile(zip_path) as zf:
    zf.extractall("/content/fantabrain-llm")

adapter = Path("models/adapters/qwen25-3b-fantabrain-sft-v2")
assert (adapter / "adapter_model.safetensors").exists()
assert (adapter / "adapter_config.json").exists()
print("adapter ok:", adapter)
```

## Guarded Pagella v2

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v2 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4 \
  --load-in-4bit \
  --torch-dtype float16 \
  --prompt-guard mode_fence_v1
```

## Audit

```bash
python scripts/audit_predictions.py \
  --predictions reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0/predictions.jsonl \
  --output-dir reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0
```

Use `--fail-on-hard-gates` when you want the cell to fail on hard violations.

## Download Report

```python
# Scarica report guarded v2.
from google.colab import files

%cd /content/fantabrain-llm
!zip -r qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0.zip \
  reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0

files.download("qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0.zip")
```
````

- [ ] **Step 2: Add README commands**

Insert after the existing Pagella adapter examples in `README.md`:

````markdown
Pagella guarded con adapter Qwen v2:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v2 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4 \
  --load-in-4bit \
  --torch-dtype float16 \
  --prompt-guard mode_fence_v1
```

Audit deterministic della run:

```bash
python scripts/audit_predictions.py \
  --predictions reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0/predictions.jsonl \
  --output-dir reports/runs/qwen25-3b-fantabrain-sft-v2-mode-fence-v1-pagella-v0
```
````

- [ ] **Step 3: Commit Task 5**

Run:

```bash
git add docs/runbooks/qwen25-v2-eval-first-guard.md README.md
git commit -m "docs: add v2 eval-first guard runbook"
```

---

### Task 6: Full Verification

**Files:**
- Verify all files changed in Tasks 1-5.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
pytest tests/test_prompt_guards.py tests/test_prediction_audit.py tests/test_generate_predictions_cli.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```bash
pytest -q
```

Expected: full suite passes.

- [ ] **Step 3: Run local echo smoke with prompt guard**

Run:

```bash
python scripts/generate_predictions.py \
  --provider echo \
  --model echo-baseline \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name echo-mode-fence-v1-smoke \
  --prompt-guard mode_fence_v1
```

Expected: `reports/runs/echo-mode-fence-v1-smoke/summary.json` exists and contains `"prompt_guard": "mode_fence_v1"`.

- [ ] **Step 4: Run local prediction audit on echo smoke**

Run:

```bash
python scripts/audit_predictions.py \
  --predictions reports/runs/echo-mode-fence-v1-smoke/predictions.jsonl \
  --output-dir reports/runs/echo-mode-fence-v1-smoke
```

Expected: `prediction_audit.json` and `prediction_audit.md` are written. This smoke verifies report mechanics, not model quality.

- [ ] **Step 5: Inspect git status**

Run:

```bash
git status --short
```

Expected: only generated `reports/runs/echo-mode-fence-v1-smoke/` is untracked if reports are not ignored. Remove the smoke report if it is untracked and not wanted in the branch.

- [ ] **Step 6: Commit final cleanup if needed**

If Task 6 revealed docs or test adjustments, commit them:

```bash
git add <changed-files>
git commit -m "test: verify v2 eval-first guard pipeline"
```

If no tracked files changed, no commit is needed.

---

## Self-Review Checklist

- Spec coverage: this plan covers prompt guard injection, deterministic prediction audit, CLI wiring, Colab runbook, README commands, local echo smoke, and no-training rollback.
- No dataset expansion: no task creates Dataset v6 or modifies training configs.
- No Pagella edits: no task modifies `benchmarks/pagella_v0.jsonl`.
- Type consistency: `prompt_guard`, `mode_fence_v1`, `prediction_audit.json`, and `prediction_audit.md` names are consistent across tests, CLI, runbook, and README.
- Known risk: echo provider only echoes the last user prompt, so CLI smoke verifies guard metadata and pipeline mechanics rather than proving the final model sees the guard.
