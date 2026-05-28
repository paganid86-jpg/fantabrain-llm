# P3bis Corrective Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Dataset v4/P3bis scaffold that assembles `datasets/v2/train.jsonl` plus 20 strict corrective examples, then trains/evaluates `qwen25-3b-fantabrain-sft-v4`.

**Architecture:** Dataset v4 mirrors the v2/v3 pattern but intentionally skips Dataset v3 as a base. A new manifest declares the 20-row P3bis shape, a new assembly CLI enforces strict anti-leak validation, a new Qwen QLoRA config targets v4, and a runbook gives Colab cells for manual authoring/training.

**Tech Stack:** Python stdlib, PyYAML, pytest, existing `fantabrain_llm.dataset` helpers, Hugging Face Transformers/TRL/PEFT/bitsandbytes for Colab training.

---

## File Structure

- Create `datasets/v4/manifest.yaml`: Dataset v4 contract, P3bis block list, strict quality gates, promotion metrics.
- Create `datasets/v4/README.md`: Human-readable summary of Dataset v4, warning that v3 is not a base.
- Create `scripts/assemble_dataset_v4.py`: Assembles v2 + P3bis rows and enforces strict target vocabulary gates.
- Create `tests/test_dataset_v4_manifest.py`: Manifest-level tests for counts, balance, source, and target cases.
- Create `tests/test_assemble_dataset_v4.py`: Assembly tests for append behavior, duplicates, Pagella leakage, and mode-vocabulary rejection.
- Create `configs/sft/qwen25-3b-qlora-v4.yaml`: Colab-friendly QLoRA config for v4.
- Modify `tests/test_training_configs.py`: Add v4 config contract test.
- Create `docs/runbooks/qwen25-lora-v4.md`: Colab runner for v2 restore, P3bis authoring, v4 assembly, training, evaluation, and artifact downloads.
- Modify `README.md`: Add v4 training/evaluation commands and Dataset v4 reference.

---

### Task 1: Dataset v4 Manifest And README

**Files:**
- Create: `datasets/v4/manifest.yaml`
- Create: `datasets/v4/README.md`
- Create: `tests/test_dataset_v4_manifest.py`

- [ ] **Step 1: Write the failing manifest tests**

Create `tests/test_dataset_v4_manifest.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_v4_manifest_uses_v2_as_base_not_v3() -> None:
    manifest_path = ROOT / "datasets" / "v4" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    assert manifest["version"] == "v4"
    assert manifest["base_dataset"] == "datasets/v2/train.jsonl"
    assert manifest["train_path"] == "datasets/v4/train.jsonl"
    assert manifest["p3bis_examples"] == 20
    assert manifest["p3bis_balance"]["by_mode"] == {"mantra": 10, "classic": 10}
    assert manifest["quality_gates"]["source"] == "v4_manual"
    assert manifest["quality_gates"]["forbid_dataset_v3_as_base"] is True


def test_dataset_v4_manifest_keeps_p3bis_blocks_balanced() -> None:
    manifest_path = ROOT / "datasets" / "v4" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    block_total = sum(block["examples"] for block in manifest["p3bis_blocks"])
    assert block_total == 20

    mode_totals = {
        "mantra": sum(block["mode_split"]["mantra"] for block in manifest["p3bis_blocks"]),
        "classic": sum(block["mode_split"]["classic"] for block in manifest["p3bis_blocks"]),
    }
    assert mode_totals == {"mantra": 10, "classic": 10}

    block_names = {block["name"] for block in manifest["p3bis_blocks"]}
    assert block_names == {
        "mantra_anti_leak",
        "classic_anti_leak",
        "decision_inversion",
        "refusal_stop",
    }


def test_dataset_v4_manifest_tracks_repair_and_promotion_targets() -> None:
    manifest_path = ROOT / "datasets" / "v4" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    repair_targets = manifest["repair_targets"]
    assert set(repair_targets["mantra_cases"]) == {2, 3, 4, 10, 11, 20, 25, 27, 38}
    assert set(repair_targets["classic_cases"]) == {13, 15, 16, 21, 24, 28, 29, 30, 36, 40}
    assert set(repair_targets["preserve_signals"]) == {6, 9, 32, 34, 37}

    promotion = manifest["promotion_gates"]
    assert promotion["effective_average_gt"] == 2.69
    assert promotion["hallucination_free_gt"] == 26
    assert promotion["raw_average_min"] == 3.10
    assert promotion["case_2_no_invented_modules"] is True
    assert promotion["mantra_forbid_modificatore"] is True
    assert promotion["classic_forbid_mantra_vocabulary"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_dataset_v4_manifest.py -q
```

Expected: FAIL with `FileNotFoundError` for `datasets/v4/manifest.yaml`.

- [ ] **Step 3: Add the Dataset v4 manifest**

Create `datasets/v4/manifest.yaml`:

```yaml
version: v4
status: draft
purpose: >
  Dataset v4 adds a small P3bis corrective set on top of Dataset v2.
  It does not use Dataset v3 as input. The focus is strict cross-mode
  cleanup, decision inversion repair, and refusal-stop behavior.

base_dataset: datasets/v2/train.jsonl
train_path: datasets/v4/train.jsonl
blind_eval_path: benchmarks/pagella_v0.jsonl

p3bis_examples: 20
p3bis_balance:
  by_mode:
    mantra: 10
    classic: 10

repair_targets:
  mantra_cases: [2, 3, 4, 10, 11, 20, 25, 27, 38]
  classic_cases: [13, 15, 16, 21, 24, 28, 29, 30, 36, 40]
  preserve_signals: [6, 9, 32, 34, 37]

p3bis_blocks:
  - name: mantra_anti_leak
    path: datasets/v4/drafts/p3bis_block_001_mantra_anti_leak.jsonl
    examples: 5
    mode_split:
      mantra: 5
      classic: 0
    focus_tag: mantra_anti_leak
    target_cases: [2, 3, 10, 11, 25]
  - name: classic_anti_leak
    path: datasets/v4/drafts/p3bis_block_002_classic_anti_leak.jsonl
    examples: 5
    mode_split:
      mantra: 0
      classic: 5
    focus_tag: classic_anti_leak
    target_cases: [13, 15, 21, 28, 29]
  - name: decision_inversion
    path: datasets/v4/drafts/p3bis_block_003_decision_inversion.jsonl
    examples: 6
    mode_split:
      mantra: 3
      classic: 3
    focus_tag: decision_inversion
    target_cases: [4, 16, 20, 24, 30, 36]
  - name: refusal_stop
    path: datasets/v4/drafts/p3bis_block_004_refusal_stop.jsonl
    examples: 4
    mode_split:
      mantra: 2
      classic: 2
    focus_tag: refusal_stop
    target_cases: [38, 40]

answer_contract:
  min_words: 55
  max_words: 90
  required_opening_markers:
    - Sceglierei
    - Preferirei
    - Eviterei
    - Non posso
  max_missing_context_sentences: 1
  max_conditional_branches: 1

quality_gates:
  min_quality_score: 5
  source: v4_manual
  no_pagella_training: true
  forbid_dataset_v3_as_base: true
  forbid_eval_prompt_leakage: true
  forbid_real_player_names: true
  forbid_specific_live_facts: true
  forbid_invented_percentages: true
  forbid_specific_scores_or_votes: true
  forbid_fake_rules: true
  forbid_invented_modules: true
  forbid_cross_mode_vocabulary: true
  forbid_broken_generated_words: true
  keep_mantra_classic_same_level: true

mode_vocabulary_gates:
  mantra_forbidden_terms:
    - modificatore
  classic_forbidden_terms:
    - slot
    - codice
    - modulo
    - moduli
    - mantra
    - pc
    - dc
    - ds
    - dd
  broken_terms:
    - offENSIVO
    - sicurata
    - ruoloni
    - mojibake
    - tre quartieri

promotion_gates:
  effective_average_gt: 2.69
  hallucination_free_gt: 26
  raw_average_min: 3.10
  case_2_no_invented_modules: true
  mantra_forbid_modificatore: true
  classic_forbid_mantra_vocabulary: true
```

- [ ] **Step 4: Add Dataset v4 README**

Create `datasets/v4/README.md`:

```markdown
# Dataset v4

Dataset v4 is the P3bis corrective dataset for FantaBrain LLM.

It is assembled as:

```text
datasets/v4/train.jsonl = datasets/v2/train.jsonl + 20 P3bis examples
```

Dataset v3 is intentionally not used as a base because the v3 Pagella regressed versus v2.

## P3bis Blocks

- `mantra_anti_leak`: 5 Mantra examples, no `modificatore`.
- `classic_anti_leak`: 5 Classic examples, no Mantra vocabulary.
- `decision_inversion`: 6 examples, 3 Mantra and 3 Classic.
- `refusal_stop`: 4 examples, 2 Mantra and 2 Classic.

All P3bis rows use `source: v4_manual` and `quality_score: 5`.

Pagella v0 remains blind.
```

- [ ] **Step 5: Run tests to verify manifest passes**

Run:

```bash
python -m pytest tests/test_dataset_v4_manifest.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add datasets/v4/manifest.yaml datasets/v4/README.md tests/test_dataset_v4_manifest.py
git commit -m "feat: scaffold dataset v4 targets"
```

---

### Task 2: Dataset v4 Assembly Validation

**Files:**
- Create: `scripts/assemble_dataset_v4.py`
- Create: `tests/test_assemble_dataset_v4.py`

- [ ] **Step 1: Write failing assembly tests**

Create `tests/test_assemble_dataset_v4.py` with helper functions and tests:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.assemble_dataset_v4 import AssemblyError, assemble_dataset


SYSTEM = "Sei il coach AI privato di FantaBrain."


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def example(
    *,
    mode: str,
    task: str,
    source: str,
    user: str,
    assistant: str,
    tags: list[str],
    quality_score: int = 5,
) -> dict[str, object]:
    return {
        "mode": mode,
        "task": task,
        "source": source,
        "quality_score": quality_score,
        "tags": tags,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def write_manifest(
    path: Path,
    block_path: Path,
    *,
    examples: int = 2,
    mode_split: dict[str, int] | None = None,
    p3bis_balance: dict[str, int] | None = None,
    blind_eval_path: str = "benchmarks/pagella_v0.jsonl",
) -> None:
    expected_split = mode_split or {"mantra": 1, "classic": 1}
    payload = {
        "version": "v4",
        "base_dataset": "base.jsonl",
        "train_path": "train.jsonl",
        "blind_eval_path": blind_eval_path,
        "p3bis_examples": examples,
        "p3bis_balance": {"by_mode": p3bis_balance or {"mantra": 1, "classic": 1}},
        "p3bis_blocks": [
            {
                "name": "test_block",
                "path": str(block_path),
                "examples": examples,
                "mode_split": expected_split,
                "focus_tag": "test_focus",
                "target_cases": [4],
            }
        ],
        "quality_gates": {
            "min_quality_score": 5,
            "source": "v4_manual",
            "forbid_eval_prompt_leakage": True,
            "forbid_dataset_v3_as_base": True,
        },
        "mode_vocabulary_gates": {
            "mantra_forbidden_terms": ["modificatore"],
            "classic_forbidden_terms": ["slot", "codice", "modulo", "moduli", "mantra", "pc"],
            "broken_terms": ["offENSIVO", "sicurata", "ruoloni"],
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_assemble_dataset_appends_valid_p3bis_rows(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"

    write_jsonl(
        base_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v2_manual",
                user="Base prompt?",
                assistant="Base answer.",
                tags=["v2", "train", "mantra"],
            )
        ],
    )
    write_jsonl(
        block_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v4_manual",
                user="P3bis Mantra prompt?",
                assistant="Sceglierei il modulo gia coperto, usando solo i ruoli indicati.",
                tags=["v4", "train", "mantra", "lineup_advice", "test_focus"],
            ),
            example(
                mode="classic",
                task="rules_explanation",
                source="v4_manual",
                user="P3bis Classic prompt?",
                assistant="Sceglierei il reparto piu affidabile se alza il voto medio.",
                tags=["v4", "train", "classic", "rules_explanation", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path)

    summary = assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)

    assert summary == {
        "base_examples": 1,
        "p3bis_examples": 2,
        "total_examples": 3,
        "p3bis_by_mode": {"mantra": 1, "classic": 1},
    }
    assert sum(1 for _ in output_path.open("r", encoding="utf-8")) == 3


def test_assemble_dataset_rejects_mantra_modificatore_target(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"

    write_jsonl(base_path, [example(mode="classic", task="lineup_advice", source="v2_manual", user="Base?", assistant="Base.", tags=["v2", "train", "classic"])])
    write_jsonl(
        block_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v4_manual",
                user="Mantra leak?",
                assistant="Sceglierei questo perche il modificatore aiuta.",
                tags=["v4", "train", "mantra", "lineup_advice", "test_focus"],
            ),
            example(
                mode="classic",
                task="lineup_advice",
                source="v4_manual",
                user="Classic clean?",
                assistant="Sceglierei il reparto con piu titolari sicuri.",
                tags=["v4", "train", "classic", "lineup_advice", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="forbidden Mantra term"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_classic_mantra_vocabulary(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"

    write_jsonl(base_path, [example(mode="mantra", task="lineup_advice", source="v2_manual", user="Base?", assistant="Base.", tags=["v2", "train", "mantra"])])
    write_jsonl(
        block_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v4_manual",
                user="Mantra clean?",
                assistant="Sceglierei la copertura se protegge il ruolo raro.",
                tags=["v4", "train", "mantra", "lineup_advice", "test_focus"],
            ),
            example(
                mode="classic",
                task="trade_advice",
                source="v4_manual",
                user="Classic leak?",
                assistant="Sceglierei questo slot perche il codice e piu flessibile.",
                tags=["v4", "train", "classic", "trade_advice", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="forbidden Classic term"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_broken_terms(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"

    write_jsonl(base_path, [example(mode="classic", task="lineup_advice", source="v2_manual", user="Base?", assistant="Base.", tags=["v2", "train", "classic"])])
    write_jsonl(
        block_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v4_manual",
                user="Mantra broken?",
                assistant="Sceglierei il talento offENSIVO solo se coperto.",
                tags=["v4", "train", "mantra", "lineup_advice", "test_focus"],
            ),
            example(
                mode="classic",
                task="lineup_advice",
                source="v4_manual",
                user="Classic clean?",
                assistant="Sceglierei il reparto piu stabile.",
                tags=["v4", "train", "classic", "lineup_advice", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="broken generated term"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)
```

- [ ] **Step 2: Run tests to verify import fails**

Run:

```bash
python -m pytest tests/test_assemble_dataset_v4.py -q
```

Expected: FAIL with `ModuleNotFoundError` or import error for `scripts.assemble_dataset_v4`.

- [ ] **Step 3: Implement Dataset v4 assembler**

Create `scripts/assemble_dataset_v4.py` by adapting `scripts/assemble_dataset_v3.py` and adding target vocabulary gates:

```python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.dataset import DatasetError, load_examples, to_sft_record, write_jsonl  # noqa: E402


class AssemblyError(ValueError):
    """Raised when Dataset v4 cannot be assembled safely."""


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise AssemblyError("Install PyYAML or run `python -m pip install -e .[dev]`.") from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise AssemblyError(f"Manifest must be a YAML object: {path}")
    return payload


def user_prompt(record: dict[str, object]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise AssemblyError("messages must be a list")
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise AssemblyError("example must include a user prompt")


def assistant_text(record: dict[str, object]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise AssemblyError("messages must be a list")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise AssemblyError("example must include an assistant target")


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def records_from_examples(path: Path) -> list[dict[str, object]]:
    try:
        return [to_sft_record(example) for example in load_examples(path)]
    except DatasetError as exc:
        raise AssemblyError(str(exc)) from exc


def load_pagella_prompts(eval_path: Path, *, required: bool) -> set[str]:
    if not eval_path.exists():
        if required:
            raise AssemblyError(f"Pagella eval file not found: {eval_path}")
        return set()
    return {normalize_prompt(user_prompt(record)) for record in records_from_examples(eval_path)}


def contains_forbidden_term(text: str, term: str) -> bool:
    lowered = text.lower()
    escaped = re.escape(term.lower())
    if term.lower() in {"pc", "dc", "ds", "dd"}:
        return re.search(rf"(?<![a-z]){escaped}(?![a-z])", lowered) is not None
    return term.lower() in lowered


def validate_target_vocabulary(
    *,
    mode: str,
    text: str,
    path: Path,
    index: int,
    gates: dict[str, object],
) -> None:
    prefix = f"{path}:{index}"
    mantra_terms = [str(term) for term in gates.get("mantra_forbidden_terms", [])]
    classic_terms = [str(term) for term in gates.get("classic_forbidden_terms", [])]
    broken_terms = [str(term) for term in gates.get("broken_terms", [])]

    for term in broken_terms:
        if contains_forbidden_term(text, term):
            raise AssemblyError(f"{prefix}: broken generated term {term!r}")

    if mode == "mantra":
        for term in mantra_terms:
            if contains_forbidden_term(text, term):
                raise AssemblyError(f"{prefix}: forbidden Mantra term {term!r}")

    if mode == "classic":
        for term in classic_terms:
            if contains_forbidden_term(text, term):
                raise AssemblyError(f"{prefix}: forbidden Classic term {term!r}")


def validate_p3bis_record(
    *,
    record: dict[str, object],
    path: Path,
    index: int,
    min_quality: int,
    required_source: str,
    focus_tag: str,
    forbidden_prompts: set[str],
    vocabulary_gates: dict[str, object],
) -> None:
    prefix = f"{path}:{index}"
    if record.get("source") != required_source:
        raise AssemblyError(f"{prefix}: source must be {required_source}")
    quality_score = record.get("quality_score")
    if not isinstance(quality_score, int) or quality_score < min_quality:
        raise AssemblyError(f"{prefix}: quality_score must be at least {min_quality}")

    mode = record.get("mode")
    task = record.get("task")
    tags = record.get("tags")
    if mode not in {"mantra", "classic"}:
        raise AssemblyError(f"{prefix}: invalid mode {mode!r}")
    if not isinstance(task, str) or not task:
        raise AssemblyError(f"{prefix}: task is required")
    if not isinstance(tags, list):
        raise AssemblyError(f"{prefix}: tags must be a list")

    required_tags = {"v4", "train", str(mode), str(task), focus_tag}
    missing_tags = sorted(required_tags - {str(tag) for tag in tags})
    if missing_tags:
        raise AssemblyError(f"{prefix}: missing tags: {', '.join(missing_tags)}")

    prompt = normalize_prompt(user_prompt(record))
    if prompt in forbidden_prompts:
        raise AssemblyError(f"{prefix}: P3bis prompt matches a pagella prompt")

    validate_target_vocabulary(
        mode=str(mode),
        text=assistant_text(record),
        path=path,
        index=index,
        gates=vocabulary_gates,
    )


def validate_unique_prompts(records: list[dict[str, object]]) -> None:
    seen: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        prompt = normalize_prompt(user_prompt(record))
        if prompt in seen:
            raise AssemblyError(f"Duplicate user prompt at rows {seen[prompt]} and {index}")
        seen[prompt] = index


def resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    root_candidate = ROOT / candidate
    if root_candidate.exists():
        return root_candidate
    return manifest_path.parent / candidate


def count_modes(records: list[dict[str, object]], expected_modes: dict[str, int]) -> dict[str, int]:
    counter = Counter(str(record["mode"]) for record in records)
    return {mode: counter.get(mode, 0) for mode in expected_modes}


def assemble_dataset(base_path: Path, manifest_path: Path, output_path: Path) -> dict[str, object]:
    manifest = load_yaml(manifest_path)
    if bool(manifest.get("quality_gates", {}).get("forbid_dataset_v3_as_base", False)):
        normalized_base = str(base_path).replace("\\", "/")
        if "datasets/v3/" in normalized_base:
            raise AssemblyError("Dataset v4 must not use Dataset v3 as base")

    base_records = records_from_examples(base_path)
    eval_path = ROOT / manifest.get("blind_eval_path", "benchmarks/pagella_v0.jsonl")

    quality_gates = manifest.get("quality_gates", {})
    vocabulary_gates = manifest.get("mode_vocabulary_gates", {})
    min_quality = int(quality_gates.get("min_quality_score", 5))
    required_source = str(quality_gates.get("source", "v4_manual"))
    forbidden_prompts = load_pagella_prompts(
        eval_path,
        required=bool(quality_gates.get("forbid_eval_prompt_leakage", False)),
    )

    p3bis_records: list[dict[str, object]] = []
    for block in manifest.get("p3bis_blocks", []):
        block_path = resolve_manifest_path(manifest_path, str(block["path"]))
        expected_examples = int(block["examples"])
        expected_split = block["mode_split"]
        focus_tag = str(block["focus_tag"])

        records = records_from_examples(block_path)
        if len(records) != expected_examples:
            raise AssemblyError(f"{block_path}: expected {expected_examples} examples, got {len(records)}")

        split = count_modes(records, expected_split)
        if split != expected_split:
            raise AssemblyError(f"{block_path}: expected mode split {expected_split}, got {split}")

        for index, record in enumerate(records, start=1):
            validate_p3bis_record(
                record=record,
                path=block_path,
                index=index,
                min_quality=min_quality,
                required_source=required_source,
                focus_tag=focus_tag,
                forbidden_prompts=forbidden_prompts,
                vocabulary_gates=vocabulary_gates,
            )

        p3bis_records.extend(records)

    expected_total = int(manifest["p3bis_examples"])
    if len(p3bis_records) != expected_total:
        raise AssemblyError(f"expected {expected_total} P3bis examples, got {len(p3bis_records)}")

    expected_by_mode = manifest["p3bis_balance"]["by_mode"]
    p3bis_by_mode = count_modes(p3bis_records, expected_by_mode)
    if p3bis_by_mode != expected_by_mode:
        raise AssemblyError(f"expected P3bis mode split {expected_by_mode}, got {p3bis_by_mode}")

    output_records = [*base_records, *p3bis_records]
    validate_unique_prompts(output_records)
    write_jsonl(output_path, output_records)

    return {
        "base_examples": len(base_records),
        "p3bis_examples": len(p3bis_records),
        "total_examples": len(output_records),
        "p3bis_by_mode": p3bis_by_mode,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble and validate Dataset v4.")
    parser.add_argument("--base", required=True, help="Final Dataset v2 JSONL path.")
    parser.add_argument("--manifest", required=True, help="Dataset v4 manifest path.")
    parser.add_argument("--output", required=True, help="Output Dataset v4 train JSONL path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = assemble_dataset(
            base_path=Path(args.base),
            manifest_path=Path(args.manifest),
            output_path=Path(args.output),
        )
    except AssemblyError as exc:
        print(f"Assembly error: {exc}", file=sys.stderr)
        return 1

    print("Dataset v4 assembled")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run assembly tests**

Run:

```bash
python -m pytest tests/test_assemble_dataset_v4.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Run v3/v4 assembly tests together**

Run:

```bash
python -m pytest tests/test_assemble_dataset_v3.py tests/test_assemble_dataset_v4.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add scripts/assemble_dataset_v4.py tests/test_assemble_dataset_v4.py
git commit -m "feat: add dataset v4 assembly validation"
```

---

### Task 3: Qwen v4 Training Config

**Files:**
- Create: `configs/sft/qwen25-3b-qlora-v4.yaml`
- Modify: `tests/test_training_configs.py`

- [ ] **Step 1: Add failing v4 config test**

Append this test to `tests/test_training_configs.py`:

```python
def test_qwen25_lora_v4_config_points_to_dataset_v4() -> None:
    config_path = ROOT / "configs" / "sft" / "qwen25-3b-qlora-v4.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["data"]["train_path"] == "datasets/v4/train.jsonl"
    assert config["data"].get("eval_path") in (None, "")
    assert "pagella" not in config["data"]["train_path"]
    assert "datasets/v3" not in config["data"]["train_path"]
    assert config["training"]["output_dir"] == "models/adapters/qwen25-3b-fantabrain-sft-v4"
    assert config["training"]["num_train_epochs"] == 2
    assert config["model"]["load_in_4bit"] is True
    assert config["model"]["torch_dtype"] == "float16"
    assert config["model"]["low_cpu_mem_usage"] is True
    assert config["training"]["bf16"] is False
    assert config["training"]["fp16"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_training_configs.py::test_qwen25_lora_v4_config_points_to_dataset_v4 -q
```

Expected: FAIL with `FileNotFoundError` for the v4 config.

- [ ] **Step 3: Add v4 config**

Create `configs/sft/qwen25-3b-qlora-v4.yaml`:

```yaml
project:
  name: fantabrain-llm
  run_name: qwen25-3b-fantabrain-sft-v4

data:
  train_path: datasets/v4/train.jsonl
  eval_path:

model:
  base_model: Qwen/Qwen2.5-3B-Instruct
  trust_remote_code: false
  load_in_4bit: true
  torch_dtype: float16
  device_map: auto
  bnb_4bit_quant_type: nf4
  use_nested_quant: true
  low_cpu_mem_usage: true

training:
  output_dir: models/adapters/qwen25-3b-fantabrain-sft-v4
  max_length: 2048
  num_train_epochs: 2
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.00018
  warmup_ratio: 0.03
  logging_steps: 5
  save_steps: 50
  eval_steps: 50
  eval_strategy: "no"
  packing: false
  assistant_only_loss: false
  report_to: none
  seed: 42
  bf16: false
  fp16: false
  gradient_checkpointing: false
  max_grad_norm: 1.0

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

- [ ] **Step 4: Run config tests**

Run:

```bash
python -m pytest tests/test_training_configs.py -q
```

Expected: all training config tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add configs/sft/qwen25-3b-qlora-v4.yaml tests/test_training_configs.py
git commit -m "feat: add qwen v4 training config"
```

---

### Task 4: Qwen v4 Runbook And README

**Files:**
- Create: `docs/runbooks/qwen25-lora-v4.md`
- Modify: `README.md`

- [ ] **Step 1: Add v4 runbook**

Create `docs/runbooks/qwen25-lora-v4.md` with these sections:

```markdown
# Qwen2.5 3B LoRA v4 Runbook

## Goal

Train `qwen25-3b-fantabrain-sft-v4` from Dataset v4 and evaluate it on Pagella v0.

Dataset v4 is assembled from `datasets/v2/train.jsonl` plus 20 P3bis corrective examples. Dataset v3 is intentionally not used as a base.

## 1. Setup GPU Check

```python
!nvidia-smi

import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
```

## 2. Clone Or Update Repo

```bash
%%bash
set -euo pipefail

REPO_URL="https://${GH_TOKEN}@github.com/paganid86-jpg/fantabrain-llm.git"

if [ ! -d fantabrain-llm/.git ]; then
  git clone "$REPO_URL" fantabrain-llm
fi

cd fantabrain-llm
git fetch origin codex/p1-dataset-v1
git switch codex/p1-dataset-v1
git pull --ff-only origin codex/p1-dataset-v1
git status -sb
```

## 3. Install Dependencies

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
python -m pip install -U "bitsandbytes>=0.46.1"
```

## 4. Restore Dataset v2

Upload `fantabrain-dataset-v2-280.zip` if `datasets/v2/train.jsonl` is missing.

```python
from google.colab import files
from pathlib import Path
import shutil
import zipfile

%cd /content/fantabrain-llm

target = Path("datasets/v2/train.jsonl")
if not target.exists():
    uploaded = files.upload()
    zip_name = next(iter(uploaded))
    tmp = Path("/content/fantabrain-upload-v2")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_name) as archive:
        archive.extractall(tmp)
    source = next(tmp.rglob("datasets/v2/train.jsonl"))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)

print("v2 examples:", sum(1 for _ in target.open(encoding="utf-8")))
```

Expected: `v2 examples: 280`.

## 5. Author P3bis Draft Blocks

Create:

```text
datasets/v4/drafts/p3bis_block_001_mantra_anti_leak.jsonl
datasets/v4/drafts/p3bis_block_002_classic_anti_leak.jsonl
datasets/v4/drafts/p3bis_block_003_decision_inversion.jsonl
datasets/v4/drafts/p3bis_block_004_refusal_stop.jsonl
```

All rows use `source: v4_manual`, `quality_score: 5`, and the tags declared in `datasets/v4/manifest.yaml`.

## 6. Assemble Dataset v4

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/assemble_dataset_v4.py \
  --base datasets/v2/train.jsonl \
  --manifest datasets/v4/manifest.yaml \
  --output datasets/v4/train.jsonl
```

Expected:

```text
Dataset v4 assembled
```

## 7. Audit Dataset v4

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("datasets/v4/train.jsonl")
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
p3bis = [row for row in rows if row.get("source") == "v4_manual"]

print("examples:", len(rows))
print("by mode:", Counter(row.get("mode") for row in rows))
print("by source:", Counter(row.get("source") for row in rows))
print("p3bis rows:", len(p3bis))
print("p3bis by mode:", Counter(row.get("mode") for row in p3bis))
print("min quality:", min(row.get("quality_score", 0) for row in rows))
PY
```

Expected:

```text
examples: 300
by mode: Counter({'mantra': 150, 'classic': 150})
p3bis rows: 20
p3bis by mode: Counter({'mantra': 10, 'classic': 10})
min quality: 4
```

## 8. Download Dataset v4

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
zip -r fantabrain-dataset-v4-300.zip datasets/v4 datasets/v2/train.jsonl datasets/v2/manifest.yaml
```

```python
from google.colab import files

files.download("fantabrain-dataset-v4-300.zip")
```

## 9. Train

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v4.yaml
```

Expected final line:

```text
Adapter saved to models/adapters/qwen25-3b-fantabrain-sft-v4
```

## 10. Download Adapter

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
zip -r qwen25-3b-fantabrain-sft-v4-adapter.zip models/adapters/qwen25-3b-fantabrain-sft-v4
```

```python
from google.colab import files

files.download("qwen25-3b-fantabrain-sft-v4-adapter.zip")
```

## 11. Evaluate

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v4 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v4-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

## 12. Verify And Download Pagella

```bash
%%bash
set -euo pipefail

cd fantabrain-llm
head -n 40 reports/runs/qwen25-3b-fantabrain-sft-v4-pagella-v0/summary.json
wc -l reports/runs/qwen25-3b-fantabrain-sft-v4-pagella-v0/predictions.jsonl
zip -r qwen25-3b-fantabrain-sft-v4-pagella-v0.zip reports/runs/qwen25-3b-fantabrain-sft-v4-pagella-v0
```

```python
from google.colab import files

files.download("qwen25-3b-fantabrain-sft-v4-pagella-v0.zip")
```
```

- [ ] **Step 2: Update README v4 training command**

Add after the v3 training section in `README.md`:

```markdown
Forgia Qwen v4, solo dopo aver completato `datasets/v4/train.jsonl`:

```bash
python -m pip install -e ".[train]"
python -m pip install -U "bitsandbytes>=0.46.1"
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v4.yaml
```

Il runbook operativo e `docs/runbooks/qwen25-lora-v4.md`.
```

- [ ] **Step 3: Update README v4 evaluation command**

Add after the v3 Pagella command in `README.md`:

```markdown
Pagella con adapter Qwen v4:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v4 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v4-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```
```

- [ ] **Step 4: Update README Dataset list**

Replace:

```markdown
Dataset v3 e descritto in `datasets/v3/manifest.yaml` e nel runbook `docs/runbooks/qwen25-lora-v3.md`.
```

with:

```markdown
Dataset v3 e descritto in `datasets/v3/manifest.yaml` e nel runbook `docs/runbooks/qwen25-lora-v3.md`.
Dataset v4 e descritto in `datasets/v4/manifest.yaml` e nel runbook `docs/runbooks/qwen25-lora-v4.md`.
```

- [ ] **Step 5: Run docs-adjacent smoke checks**

Run:

```bash
python -m pytest tests/test_dataset_v4_manifest.py tests/test_training_configs.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add docs/runbooks/qwen25-lora-v4.md README.md
git commit -m "docs: add qwen v4 runbook"
```

---

### Task 5: Full Verification

**Files:**
- No new files.
- Verify all files from Tasks 1-4.

- [ ] **Step 1: Run targeted v4 tests**

Run:

```bash
python -m pytest \
  tests/test_dataset_v4_manifest.py \
  tests/test_assemble_dataset_v4.py \
  tests/test_training_configs.py \
  -q
```

Expected: all targeted tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected: full suite passes.

- [ ] **Step 3: Inspect git diff**

Run:

```bash
git status -sb
git log --oneline -5
```

Expected:

- working tree clean;
- branch ahead by the P3bis spec commit plus four scaffold commits;
- latest commits include:
  - `feat: scaffold dataset v4 targets`
  - `feat: add dataset v4 assembly validation`
  - `feat: add qwen v4 training config`
  - `docs: add qwen v4 runbook`

- [ ] **Step 4: Push branch**

Run:

```bash
git push origin codex/p1-dataset-v1
```

Expected: push succeeds.

- [ ] **Step 5: Update llm-memory**

Update `C:/Users/DantePagani/llm-memory/wiki/projects/fantabrain-llm/project-overview.md` with:

```markdown
## P3bis Repo Scaffold Pushed - 2026-05-28

Codex implemented and pushed the P3bis/Dataset v4 scaffold on branch `codex/p1-dataset-v1`. Dataset v4 is defined as Dataset v2 plus 20 P3bis examples, with Dataset v3 excluded from the training chain. The scaffold includes `datasets/v4/manifest.yaml`, `scripts/assemble_dataset_v4.py`, `configs/sft/qwen25-3b-qlora-v4.yaml`, `docs/runbooks/qwen25-lora-v4.md`, README updates, and tests.

Next step: use the v4 runbook in Colab to author the four P3bis draft blocks manually, assemble `datasets/v4/train.jsonl`, train `qwen25-3b-fantabrain-sft-v4`, and evaluate it on Pagella v0 against v2.
```

Run:

```bash
git -C "C:/Users/DantePagani/llm-memory" status -sb
```

Expected: `project-overview.md` modified; unrelated memory changes may already exist and should not be reverted.

---

## Self-Review Checklist

- Spec coverage: Tasks cover Dataset v4 manifest, strict assembly gates, training config, runbook, README, verification, push, and memory update.
- Scope: This plan stops at repo scaffold and Colab instructions. It does not author the 20 examples inside the repo and does not train locally.
- Test strategy: Manifest tests prove shape, assembly tests prove strict vocabulary gates, config tests prove v4 training points to Dataset v4 and not Pagella/v3.
- Promotion: Actual v4 promotion is deferred until the user runs Colab training/eval and Codex scores Pagella v4 against v2.
