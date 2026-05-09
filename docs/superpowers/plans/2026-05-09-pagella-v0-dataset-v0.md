# Pagella v0 And Dataset v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the versioned Pagella v0 and Dataset v0 files for FantaBrain LLM with exact 50/50 Mantra and Classic coverage.

**Architecture:** Keep the authored source-of-truth data in versioned JSONL files, then add a lightweight audit layer that verifies schema validity, task distribution, and train/eval separation. Use the existing dataset loader for row-level validation and a new v0 audit module for aggregate guarantees.

**Tech Stack:** Python 3.11 stdlib, existing `fantabrain_llm` package, JSONL source data, pytest for local verification.

---

## File Structure

- Create `datasets/v0/train.jsonl`: 120 supervised training examples, 60 Mantra and 60 Classic.
- Create `benchmarks/pagella_v0.jsonl`: 40 blind eval examples, 20 Mantra and 20 Classic.
- Create `src/fantabrain_llm/audit.py`: aggregate validation for expected counts, task/mode matrix, duplicate prompts, and train/eval leakage.
- Create `scripts/audit_v0.py`: CLI wrapper around the audit module.
- Create `tests/test_v0_audit.py`: unit tests for count checks and leakage detection.
- Modify `README.md`: add the canonical v0 audit and pagella commands.
- Modify `docs/superpowers/plans/2026-05-09-pagella-v0-dataset-v0.md`: mark tasks as completed while executing.

## V0 Target Matrix

Dataset v0:

| Task area | Task id | Total | Mantra | Classic |
| --- | --- | ---: | ---: | ---: |
| Lineup advice | `lineup_advice` | 24 | 12 | 12 |
| Auction advice | `auction_advice` | 24 | 12 | 12 |
| Trade / market advice | `trade_advice` | 24 | 12 | 12 |
| Rules / tactical explanation | `rules_explanation` | 18 | 9 | 9 |
| Risk management | `risk_management` | 18 | 9 | 9 |
| Refusal / grounding | `refusal_grounding` | 12 | 6 | 6 |
| **Total** |  | **120** | **60** | **60** |

Pagella v0:

| Task area | Task id | Total | Mantra | Classic |
| --- | --- | ---: | ---: | ---: |
| Lineup advice | `lineup_advice` | 8 | 4 | 4 |
| Auction advice | `auction_advice` | 8 | 4 | 4 |
| Trade / market advice | `trade_advice` | 8 | 4 | 4 |
| Rules / tactical explanation | `rules_explanation` | 6 | 3 | 3 |
| Risk management | `risk_management` | 6 | 3 | 3 |
| Refusal / grounding | `refusal_grounding` | 4 | 2 | 2 |
| **Total** |  | **40** | **20** | **20** |

## Authorship Rules

Every authored row must:

- use `source: "v0_manual"`;
- use `quality_score: 5` for gold-standard examples or `quality_score: 4` for useful but slightly less comprehensive examples;
- include tags for `v0`, mode, task id, and one skill label such as `formation`, `budget`, `market`, `grounding`, `modificatore`, `slot`, `captaincy`, or `bench`;
- avoid real private user data and credentials;
- avoid live facts such as current injuries or real-time votes;
- make missing-data uncertainty explicit.

The assistant answer should usually be 70-130 Italian words. Refusal/grounding answers may be shorter when the correct behavior is to ask for missing context.

---

### Task 1: Add V0 Audit Core

**Files:**
- Create: `src/fantabrain_llm/audit.py`
- Create: `scripts/audit_v0.py`
- Create: `tests/test_v0_audit.py`

- [x] **Step 1: Write failing tests for aggregate counts**

Create `tests/test_v0_audit.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fantabrain_llm.audit import AuditError, ExpectedMatrix, audit_examples, audit_train_eval_split
from fantabrain_llm.dataset import load_examples


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def row(mode: str, task: str, user: str) -> dict[str, object]:
    return {
        "mode": mode,
        "task": task,
        "source": "v0_manual",
        "quality_score": 5,
        "tags": ["v0", mode, task],
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "Risposta utile e concreta."},
        ],
    }


def test_audit_examples_accepts_expected_matrix(tmp_path: Path) -> None:
    path = tmp_path / "mini.jsonl"
    write_jsonl(
        path,
        [
            row("mantra", "lineup_advice", "Prompt mantra"),
            row("classic", "lineup_advice", "Prompt classic"),
        ],
    )
    examples = load_examples(path)

    audit_examples(
        examples,
        ExpectedMatrix(total=2, by_mode={"mantra": 1, "classic": 1}, by_task_mode={("lineup_advice", "mantra"): 1, ("lineup_advice", "classic"): 1}),
    )


def test_audit_examples_rejects_wrong_mode_count(tmp_path: Path) -> None:
    path = tmp_path / "mini.jsonl"
    write_jsonl(path, [row("mantra", "lineup_advice", "Prompt mantra")])
    examples = load_examples(path)

    with pytest.raises(AuditError, match="mode classic"):
        audit_examples(
            examples,
            ExpectedMatrix(total=2, by_mode={"mantra": 1, "classic": 1}, by_task_mode={("lineup_advice", "mantra"): 1, ("lineup_advice", "classic"): 1}),
        )


def test_audit_train_eval_split_rejects_duplicate_user_prompt(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(train_path, [row("mantra", "lineup_advice", "Stesso prompt")])
    write_jsonl(eval_path, [row("classic", "lineup_advice", "Stesso prompt")])

    with pytest.raises(AuditError, match="leakage"):
        audit_train_eval_split(load_examples(train_path), load_examples(eval_path))
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_v0_audit.py -v
```

Expected: FAIL because `fantabrain_llm.audit` does not exist.

- [x] **Step 3: Implement `src/fantabrain_llm/audit.py`**

Use this implementation:

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from fantabrain_llm.schema import TrainingExample


class AuditError(ValueError):
    """Raised when a dataset is valid row-by-row but wrong in aggregate."""


@dataclass(frozen=True)
class ExpectedMatrix:
    total: int
    by_mode: dict[str, int]
    by_task_mode: dict[tuple[str, str], int]


DATASET_V0_MATRIX = ExpectedMatrix(
    total=120,
    by_mode={"mantra": 60, "classic": 60},
    by_task_mode={
        ("lineup_advice", "mantra"): 12,
        ("lineup_advice", "classic"): 12,
        ("auction_advice", "mantra"): 12,
        ("auction_advice", "classic"): 12,
        ("trade_advice", "mantra"): 12,
        ("trade_advice", "classic"): 12,
        ("rules_explanation", "mantra"): 9,
        ("rules_explanation", "classic"): 9,
        ("risk_management", "mantra"): 9,
        ("risk_management", "classic"): 9,
        ("refusal_grounding", "mantra"): 6,
        ("refusal_grounding", "classic"): 6,
    },
)

PAGELLA_V0_MATRIX = ExpectedMatrix(
    total=40,
    by_mode={"mantra": 20, "classic": 20},
    by_task_mode={
        ("lineup_advice", "mantra"): 4,
        ("lineup_advice", "classic"): 4,
        ("auction_advice", "mantra"): 4,
        ("auction_advice", "classic"): 4,
        ("trade_advice", "mantra"): 4,
        ("trade_advice", "classic"): 4,
        ("rules_explanation", "mantra"): 3,
        ("rules_explanation", "classic"): 3,
        ("risk_management", "mantra"): 3,
        ("risk_management", "classic"): 3,
        ("refusal_grounding", "mantra"): 2,
        ("refusal_grounding", "classic"): 2,
    },
)


def _last_user_prompt(example: TrainingExample) -> str:
    for message in reversed(example.messages):
        if message.role == "user":
            return " ".join(message.content.lower().split())
    raise AuditError("example has no user prompt")


def audit_examples(examples: list[TrainingExample], expected: ExpectedMatrix) -> None:
    if len(examples) != expected.total:
        raise AuditError(f"expected {expected.total} examples, got {len(examples)}")

    mode_counts = Counter(example.mode for example in examples)
    for mode, expected_count in expected.by_mode.items():
        actual = mode_counts[mode]
        if actual != expected_count:
            raise AuditError(f"mode {mode}: expected {expected_count}, got {actual}")

    task_mode_counts = Counter((example.task, example.mode) for example in examples)
    for task_mode, expected_count in expected.by_task_mode.items():
        actual = task_mode_counts[task_mode]
        if actual != expected_count:
            task, mode = task_mode
            raise AuditError(f"{task}/{mode}: expected {expected_count}, got {actual}")

    prompts = [_last_user_prompt(example) for example in examples]
    duplicates = [prompt for prompt, count in Counter(prompts).items() if count > 1]
    if duplicates:
        raise AuditError(f"duplicate user prompts inside dataset: {duplicates[:3]}")


def audit_train_eval_split(
    train_examples: list[TrainingExample],
    eval_examples: list[TrainingExample],
) -> None:
    train_prompts = {_last_user_prompt(example) for example in train_examples}
    eval_prompts = {_last_user_prompt(example) for example in eval_examples}
    overlap = sorted(train_prompts & eval_prompts)
    if overlap:
        raise AuditError(f"train/eval leakage detected for prompts: {overlap[:3]}")
```

- [x] **Step 4: Implement `scripts/audit_v0.py`**

Use this implementation:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.audit import (  # noqa: E402
    DATASET_V0_MATRIX,
    PAGELLA_V0_MATRIX,
    AuditError,
    audit_examples,
    audit_train_eval_split,
)
from fantabrain_llm.dataset import DatasetError, load_examples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FantaBrain Dataset v0 and Pagella v0.")
    parser.add_argument("--train", default="datasets/v0/train.jsonl")
    parser.add_argument("--pagella", default="benchmarks/pagella_v0.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        train_examples = load_examples(args.train)
        pagella_examples = load_examples(args.pagella)
        audit_examples(train_examples, DATASET_V0_MATRIX)
        audit_examples(pagella_examples, PAGELLA_V0_MATRIX)
        audit_train_eval_split(train_examples, pagella_examples)
    except (DatasetError, AuditError) as exc:
        print(f"Audit error: {exc}", file=sys.stderr)
        return 1

    print("FantaBrain v0 audit passed")
    print(f"  train:   {args.train} ({len(train_examples)} examples)")
    print(f"  pagella: {args.pagella} ({len(pagella_examples)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 5: Run tests to verify audit core passes**

Run:

```powershell
python -m pytest tests/test_v0_audit.py -v
```

Expected: PASS.

- [x] **Step 6: Commit audit tooling**

Run:

```powershell
git add src/fantabrain_llm/audit.py scripts/audit_v0.py tests/test_v0_audit.py
git commit -m "feat: add v0 dataset audit tooling"
```

Expected: one commit with audit code and tests.

---

### Task 2: Add Pagella v0 Source Data

**Files:**
- Create: `benchmarks/pagella_v0.jsonl`
- Modify: `README.md`

- [x] **Step 1: Create the benchmarks directory**

Run:

```powershell
New-Item -ItemType Directory -Force benchmarks
```

Expected: `benchmarks/` exists.

- [x] **Step 2: Author 40 blind pagella rows**

Create `benchmarks/pagella_v0.jsonl` with exactly 40 JSONL rows. Use the shared system prompt from `src/fantabrain_llm/prompts.py`.

The required row matrix is:

```text
lineup_advice/mantra: 4
lineup_advice/classic: 4
auction_advice/mantra: 4
auction_advice/classic: 4
trade_advice/mantra: 4
trade_advice/classic: 4
rules_explanation/mantra: 3
rules_explanation/classic: 3
risk_management/mantra: 3
risk_management/classic: 3
refusal_grounding/mantra: 2
refusal_grounding/classic: 2
```

Use this exact row shape for every entry:

```json
{"mode":"mantra","task":"lineup_advice","source":"v0_manual","quality_score":5,"tags":["v0","pagella","mantra","lineup_advice","slot"],"messages":[{"role":"system","content":"Sei il coach AI privato di FantaBrain. Rispondi in italiano, distingui sempre Fantacalcio Mantra e Fantacalcio Classic, dai consigli concreti e segnala i dati mancanti invece di inventare statistiche."},{"role":"user","content":"Modalita Mantra. Ho due moduli possibili e un dubbio sugli slot: meglio proteggere un ruolo raro o spingere con il talento offensivo?"},{"role":"assistant","content":"In Mantra parto dal ruolo raro, non dal nome piu brillante. Se lo slot scoperto ti costringe a forzare un giocatore fuori posizione o con minutaggio fragile, il talento offensivo perde valore pratico. Sceglierei il modulo piu solido sugli incastri e poi cercherei upside negli slot flessibili. Se invece entrambi i moduli sono coperti, allora puoi spingere sul talento. Mi manca la lista completa dei ruoli, quindi la regola e: prima stabilita degli slot, poi potenziale bonus."}]}
```

All 40 prompts must be distinct from each other and from `datasets/v0/train.jsonl`.

- [x] **Step 3: Run pagella row validation**

Run:

```powershell
python scripts/run_eval.py --eval benchmarks/pagella_v0.jsonl --run-name pagella-v0-smoke
```

Expected: a Markdown report under `reports/runs/pagella-v0-smoke/manual_review.md`.

- [x] **Step 4: Update README pagella command**

Add this command to `README.md` under "Pagella manuale":

```powershell
python scripts/run_eval.py `
  --eval benchmarks/pagella_v0.jsonl `
  --run-name pagella-v0-manual-review
```

- [x] **Step 5: Commit pagella data**

Run:

```powershell
git add benchmarks/pagella_v0.jsonl README.md
git commit -m "data: add pagella v0 benchmark"
```

Expected: one commit with the versioned pagella and README update.

---

### Task 3: Add Dataset v0 Source Data

**Files:**
- Create: `datasets/v0/train.jsonl`
- Modify: `README.md`

- [x] **Step 1: Create the dataset directory**

Run:

```powershell
New-Item -ItemType Directory -Force datasets\v0
```

Expected: `datasets/v0/` exists.

- [x] **Step 2: Author 120 supervised training rows**

Create `datasets/v0/train.jsonl` with exactly 120 JSONL rows. Use the shared system prompt from `src/fantabrain_llm/prompts.py`.

The required row matrix is:

```text
lineup_advice/mantra: 12
lineup_advice/classic: 12
auction_advice/mantra: 12
auction_advice/classic: 12
trade_advice/mantra: 12
trade_advice/classic: 12
rules_explanation/mantra: 9
rules_explanation/classic: 9
risk_management/mantra: 9
risk_management/classic: 9
refusal_grounding/mantra: 6
refusal_grounding/classic: 6
```

Use this exact row shape for every entry:

```json
{"mode":"classic","task":"auction_advice","source":"v0_manual","quality_score":5,"tags":["v0","train","classic","auction_advice","budget"],"messages":[{"role":"system","content":"Sei il coach AI privato di FantaBrain. Rispondi in italiano, distingui sempre Fantacalcio Mantra e Fantacalcio Classic, dai consigli concreti e segnala i dati mancanti invece di inventare statistiche."},{"role":"user","content":"Modalita Classic. In asta ho gia due attaccanti costosi. Ha senso prendere un terzo nome premium o distribuire budget a centrocampo?"},{"role":"assistant","content":"In Classic eviterei di bloccare troppo budget sul terzo attaccante premium se il centrocampo e ancora leggero. Dopo due punte forti, il salto marginale del terzo nome spesso pesa meno di due centrocampisti da bonus o titolarita stabile. Terrei una soglia massima per il terzo attaccante e sposterei risorse su centrocampo e primi cambi. Farei eccezione solo se il terzo premium e sottoprezzo chiaro o se il regolamento premia molto il tridente."}]}
```

Training prompts must be distinct from every pagella prompt. Keep examples practical and varied, but do not use live facts, private data, or current injury/news claims.

- [x] **Step 3: Update README dataset command**

Add this command to `README.md` under "Primo giro dati":

```powershell
python scripts/prepare_dataset.py `
  --input datasets/v0/train.jsonl `
  --output data/processed/v0_train.jsonl `
  --eval-output data/eval/v0_holdout.jsonl `
  --eval-ratio 0
```

- [x] **Step 4: Commit dataset data**

Run:

```powershell
git add datasets/v0/train.jsonl README.md
git commit -m "data: add dataset v0 training examples"
```

Expected: one commit with the versioned training data and README update.

---

### Task 4: Run Full V0 Audit And Prepare Outputs

**Files:**
- Generated ignored files under `data/processed/`, `data/eval/`, and `reports/runs/`

- [x] **Step 1: Run the v0 audit CLI**

Run:

```powershell
python scripts/audit_v0.py --train datasets/v0/train.jsonl --pagella benchmarks/pagella_v0.jsonl
```

Expected:

```text
FantaBrain v0 audit passed
  train:   datasets/v0/train.jsonl (120 examples)
  pagella: benchmarks/pagella_v0.jsonl (40 examples)
```

- [x] **Step 2: Prepare v0 training file without random holdout**

Run:

```powershell
python scripts/prepare_dataset.py --input datasets/v0/train.jsonl --output data/processed/v0_train.jsonl --eval-output data/eval/v0_holdout.jsonl --eval-ratio 0
```

Expected:

```text
FantaBrain dataset prepared
  source: datasets/v0/train.jsonl
  train:  data/processed/v0_train.jsonl (120 examples)
  eval:   data/eval/v0_holdout.jsonl (0 examples)
```

- [x] **Step 3: Generate the manual pagella report**

Run:

```powershell
python scripts/run_eval.py --eval benchmarks/pagella_v0.jsonl --run-name pagella-v0-manual-review
```

Expected: `reports/runs/pagella-v0-manual-review/manual_review.md` exists and contains 40 cases.

- [x] **Step 4: Run compile check**

Run:

```powershell
python -m compileall src scripts tests
```

Expected: all files compile.

- [x] **Step 5: Run all tests**

Run:

```powershell
python -m pytest
```

Expected: all tests pass after installing dev dependencies.

---

### Task 5: Update Memory And Final Notes

**Files:**
- Modify: `C:\Users\DantePagani\llm-memory\wiki\projects\fantabrain-llm\project-overview.md`
- Modify: `C:\Users\DantePagani\llm-memory\wiki\projects\fantabrain-llm\architecture\data-model.md`

- [x] **Step 1: Update project memory**

Add a concise note that Dataset v0 and Pagella v0 exist in:

```text
datasets/v0/train.jsonl
benchmarks/pagella_v0.jsonl
```

Include the exact final counts and the latest commit hashes.

- [x] **Step 2: Run memory secret scan**

Run from `C:\Users\DantePagani\llm-memory`:

```powershell
Select-String -Path "wiki\projects\fantabrain-llm\**\*" -Pattern "API_KEY|SECRET|DATABASE_URL|BEGIN PRIVATE KEY|password" -CaseSensitive:$false
```

Expected: no actual secret values.

- [x] **Step 3: Report completion**

Final response should include:

- files created;
- counts verified by audit;
- test/compile status;
- whether pytest was unavailable or passing;
- reminder that v0 is proof-of-learning, not launch-grade.
