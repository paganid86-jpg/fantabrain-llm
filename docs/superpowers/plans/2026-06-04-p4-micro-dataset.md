# P4 Micro Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Dataset v5 as `datasets/v2/train.jsonl + 16 P4 examples`, add strict P4 leakage validation, train config, and runbook for a fresh Qwen LoRA v5 experiment.

**Architecture:** Mirror the existing Dataset v4 scaffold, but tighten validation around token-aware Classic role-code leakage, Mantra `modificatore` leakage, prompt duplication, malformed Italian, and final v5 balance. Keep notebooks as runners; reusable behavior belongs in `scripts/`, `configs/`, `tests/`, and docs.

**Tech Stack:** Python stdlib, PyYAML, pytest, existing `fantabrain_llm.dataset` helpers, Hugging Face/TRL/PEFT/bitsandbytes config conventions.

---

## File Structure

- Create `datasets/v5/manifest.yaml`: declares Dataset v5 shape, four P4 draft blocks, vocabulary gates, promotion gates, and source/quality requirements.
- Create `datasets/v5/README.md`: concise dataset purpose, block list, and authoring rules.
- Create `datasets/v5/drafts/.gitkeep`: keeps draft directory present without committing generated/manual rows yet.
- Create `scripts/assemble_dataset_v5.py`: validates P4 draft rows and writes `datasets/v5/train.jsonl` from Dataset v2 plus P4 rows.
- Create `tests/test_assemble_dataset_v5.py`: TDD coverage for happy path, Mantra forbidden vocabulary, Classic token-aware role-code leakage, duplicate train prompts, malformed words, and final balance.
- Modify `tests/test_training_configs.py`: add v5 config guard.
- Create `configs/sft/qwen25-3b-qlora-v5.yaml`: Colab-friendly QLoRA config for v5.
- Create `docs/runbooks/qwen25-lora-v5.md`: Colab workflow for authoring, assembling, training, evaluating, downloading, and scoring v5.
- Modify `README.md`: add compact v5 commands and note that v2 remains rollback.

---

### Task 1: Scaffold Dataset v5 Metadata

**Files:**
- Create: `datasets/v5/manifest.yaml`
- Create: `datasets/v5/README.md`
- Create: `datasets/v5/drafts/.gitkeep`

- [ ] **Step 1: Create Dataset v5 manifest**

Create `datasets/v5/manifest.yaml`:

```yaml
version: v5
status: draft
purpose: >
  Dataset v5 adds a 16-example P4 micro corrective set on top of Dataset v2.
  It does not use Dataset v3 or Dataset v4 as input. The focus is lexical
  leakage control, no invented modules, and clean decision-first Italian.

base_dataset: datasets/v2/train.jsonl
train_path: datasets/v5/train.jsonl
blind_eval_path: benchmarks/pagella_v0.jsonl

p4_examples: 16
p4_balance:
  by_mode:
    mantra: 8
    classic: 8

final_balance:
  by_mode:
    mantra: 148
    classic: 148

repair_targets:
  mantra_cases: [2, 3, 20, 31, 38]
  classic_cases: [14, 28, 36, 39]
  preserve_signals: [17, 19, 34, 37]

p4_blocks:
  - name: mantra_no_modificatore
    path: datasets/v5/drafts/p4_block_001_mantra_no_modificatore.jsonl
    examples: 4
    mode_split:
      mantra: 4
      classic: 0
    focus_tag: mantra_no_modificatore
    target_cases: [3, 20, 31, 38]
  - name: classic_clean_vocab
    path: datasets/v5/drafts/p4_block_002_classic_clean_vocab.jsonl
    examples: 4
    mode_split:
      mantra: 0
      classic: 4
    focus_tag: classic_clean_vocab
    target_cases: [14, 28, 36, 39]
  - name: no_invented_modules
    path: datasets/v5/drafts/p4_block_003_no_invented_modules.jsonl
    examples: 4
    mode_split:
      mantra: 4
      classic: 0
    focus_tag: no_invented_modules
    target_cases: [2]
  - name: italian_decision_clean
    path: datasets/v5/drafts/p4_block_004_italian_decision_clean.jsonl
    examples: 4
    mode_split:
      mantra: 0
      classic: 4
    focus_tag: italian_decision_clean
    target_cases: [14, 28, 36, 39]

answer_contract:
  min_words: 45
  max_words: 75
  required_opening_markers:
    - Sceglierei
    - Preferirei
    - Eviterei
    - Non posso
  max_missing_context_sentences: 1
  max_conditional_branches: 1

quality_gates:
  min_quality_score: 5
  source: v5_manual
  no_pagella_training: true
  forbid_dataset_v3_as_base: true
  forbid_dataset_v4_as_base: true
  forbid_eval_prompt_leakage: true
  forbid_train_prompt_duplicates: true
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
    - reparto
  classic_forbidden_terms:
    - slot
    - incastro
    - codice
    - mantra
  classic_forbidden_role_codes:
    - Pc
    - T
    - W
    - M
    - C
    - E
    - Dc
    - Ds
    - Dd
  broken_terms:
    - offENSIVO
    - sicurata
    - multicolore
    - punteggianza
    - malusso
    - maleducata
    - aspettarello
    - esattissimi
    - mojibake

promotion_gates:
  effective_average_gt: 2.69
  hallucination_free_gt: 26
  raw_average_min: 3.10
  case_2_no_invented_modules: true
  mantra_forbid_modificatore: true
  classic_forbid_role_code_leakage: true
  classic_cases_no_regression: [14, 28, 36, 39]
  malformed_italian_max_repeated_cases: 1
```

- [ ] **Step 2: Create Dataset v5 README**

Create `datasets/v5/README.md`:

```markdown
# Dataset v5

Dataset v5 is the P4 micro corrective dataset for FantaBrain LLM.

It is assembled as:

```text
datasets/v5/train.jsonl = datasets/v2/train.jsonl + 16 P4 examples
```

Dataset v3 and Dataset v4 are intentionally not used as a base. Qwen LoRA v2 remains the rollback baseline.

## P4 Blocks

- `mantra_no_modificatore`: 4 Mantra examples, no `modificatore` and no `reparto`.
- `classic_clean_vocab`: 4 Classic examples, no Mantra role-code language.
- `no_invented_modules`: 4 Mantra examples, use only modules named in the prompt.
- `italian_decision_clean`: 4 Classic examples, short decision-first answers with clean Italian.

All P4 rows use `source: v5_manual` and `quality_score: 5`.

Pagella v0 remains blind.
```
```

- [ ] **Step 3: Keep drafts directory**

Run:

```powershell
New-Item -ItemType Directory -Force -Path datasets\v5\drafts | Out-Null
New-Item -ItemType File -Force -Path datasets\v5\drafts\.gitkeep | Out-Null
```

Expected: directory exists and `git status -sb` shows the three new Dataset v5 scaffold files.

- [ ] **Step 4: Commit scaffold**

Run:

```bash
git add datasets/v5
git commit -m "feat: scaffold dataset v5 metadata"
```

Expected: commit succeeds.

---

### Task 2: Add Failing Assembly Tests

**Files:**
- Create: `tests/test_assemble_dataset_v5.py`

- [ ] **Step 1: Write v5 assembly tests**

Create `tests/test_assemble_dataset_v5.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.assemble_dataset_v5 import AssemblyError, assemble_dataset


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
    p4_balance: dict[str, int] | None = None,
    final_balance: dict[str, int] | None = None,
    blind_eval_path: str = "benchmarks/pagella_v0.jsonl",
) -> None:
    split = mode_split or {"mantra": 1, "classic": 1}
    payload = {
        "version": "v5",
        "base_dataset": "base.jsonl",
        "train_path": "train.jsonl",
        "blind_eval_path": blind_eval_path,
        "p4_examples": examples,
        "p4_balance": {"by_mode": p4_balance or {"mantra": 1, "classic": 1}},
        "final_balance": {"by_mode": final_balance or {"mantra": 2, "classic": 2}},
        "p4_blocks": [
            {
                "name": "test_block",
                "path": str(block_path),
                "examples": examples,
                "mode_split": split,
                "focus_tag": "test_focus",
                "target_cases": [2],
            }
        ],
        "quality_gates": {
            "min_quality_score": 5,
            "source": "v5_manual",
            "forbid_eval_prompt_leakage": True,
            "forbid_train_prompt_duplicates": True,
            "forbid_dataset_v3_as_base": True,
            "forbid_dataset_v4_as_base": True,
        },
        "mode_vocabulary_gates": {
            "mantra_forbidden_terms": ["modificatore", "reparto"],
            "classic_forbidden_terms": ["slot", "incastro", "codice", "mantra"],
            "classic_forbidden_role_codes": ["Pc", "T", "W", "M", "C", "E", "Dc", "Ds", "Dd"],
            "broken_terms": ["offENSIVO", "sicurata", "punteggianza"],
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def valid_mantra(user: str = "P4 Mantra prompt?") -> dict[str, object]:
    return example(
        mode="mantra",
        task="lineup_advice",
        source="v5_manual",
        user=user,
        assistant="Sceglierei il modulo gia coperto nei ruoli indicati. Se hai un dubbio tra talento e copertura, parto dalla stabilita dello slot raro e poi cerco bonus negli spazi flessibili. Mi manca la lista completa dei ruoli.",
        tags=["v5", "train", "mantra", "lineup_advice", "test_focus"],
    )


def valid_classic(user: str = "P4 Classic prompt?") -> dict[str, object]:
    return example(
        mode="classic",
        task="lineup_advice",
        source="v5_manual",
        user=user,
        assistant="Sceglierei il reparto con piu titolari sicuri. Se il bonus atteso e simile, proteggo voto medio e panchina prima della scommessa. Mi mancano nomi e avversari, quindi tengo una linea prudente.",
        tags=["v5", "train", "classic", "lineup_advice", "test_focus"],
    )


def test_assemble_dataset_appends_valid_p4_rows(tmp_path: Path) -> None:
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
                user="Base Mantra?",
                assistant="Base answer.",
                tags=["v2", "train", "mantra"],
            ),
            example(
                mode="classic",
                task="lineup_advice",
                source="v2_manual",
                user="Base Classic?",
                assistant="Base answer.",
                tags=["v2", "train", "classic"],
            ),
        ],
    )
    write_jsonl(block_path, [valid_mantra(), valid_classic()])
    write_manifest(manifest_path, block_path)

    summary = assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)

    assert summary == {
        "base_examples": 2,
        "p4_examples": 2,
        "total_examples": 4,
        "p4_by_mode": {"mantra": 1, "classic": 1},
        "final_by_mode": {"mantra": 2, "classic": 2},
    }
    assert sum(1 for _ in output_path.open("r", encoding="utf-8")) == 4


def test_assemble_dataset_rejects_dataset_v3_or_v4_base(tmp_path: Path) -> None:
    base_path = tmp_path / "datasets" / "v3" / "train.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Base Mantra?"), valid_classic(user="Base Classic?")])
    write_jsonl(block_path, [valid_mantra(), valid_classic()])
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="must not use Dataset v3"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_duplicate_base_prompt(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Duplicate prompt?"), valid_classic(user="Base Classic?")])
    write_jsonl(block_path, [valid_mantra(user="Duplicate prompt?"), valid_classic()])
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="Duplicate user prompt"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_mantra_forbidden_terms(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Base Mantra?"), valid_classic(user="Base Classic?")])
    bad = valid_mantra()
    bad["messages"][-1]["content"] = "Sceglierei questa strada perche il modificatore protegge il reparto."
    write_jsonl(block_path, [bad, valid_classic()])
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="forbidden Mantra term"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_classic_role_code_token_but_not_letters_inside_words(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Base Mantra?"), valid_classic(user="Base Classic?")])
    clean = valid_classic()
    clean["messages"][-1]["content"] = "Sceglierei il centrocampo stabile. Mi manca il calendario, ma in una rosa corta proteggo panchina e voto medio prima del bonus."
    leaking = valid_classic(user="Classic role leak?")
    leaking["messages"][-1]["content"] = "Sceglierei il reparto sicuro, ma la W cambia il ragionamento."
    write_jsonl(block_path, [valid_mantra(), clean])
    write_manifest(manifest_path, block_path)
    assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)

    write_jsonl(block_path, [valid_mantra(), leaking])
    with pytest.raises(AssemblyError, match="forbidden Classic role code"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_broken_terms(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Base Mantra?"), valid_classic(user="Base Classic?")])
    bad = valid_classic()
    bad["messages"][-1]["content"] = "Sceglierei la punteggianza migliore se il reparto resta coperto."
    write_jsonl(block_path, [valid_mantra(), bad])
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="broken generated term"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_final_balance_mismatch(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Base Mantra?"), valid_classic(user="Base Classic?")])
    write_jsonl(block_path, [valid_mantra(), valid_classic()])
    write_manifest(manifest_path, block_path, final_balance={"mantra": 3, "classic": 1})

    with pytest.raises(AssemblyError, match="expected final mode split"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_assemble_dataset_v5.py -q
```

Expected: collection/import fails with `ModuleNotFoundError: No module named 'scripts.assemble_dataset_v5'` or `ImportError` because `scripts/assemble_dataset_v5.py` does not exist yet.

- [ ] **Step 3: Commit failing tests**

Run:

```bash
git add tests/test_assemble_dataset_v5.py
git commit -m "test: add dataset v5 assembly expectations"
```

Expected: commit succeeds with failing tests intentionally present.

---

### Task 3: Implement Dataset v5 Assembly

**Files:**
- Create: `scripts/assemble_dataset_v5.py`

- [ ] **Step 1: Create assembly script**

Create `scripts/assemble_dataset_v5.py`:

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

from fantabrain_llm.dataset import (  # noqa: E402
    DatasetError,
    load_examples,
    to_sft_record,
    write_jsonl,
)


class AssemblyError(ValueError):
    """Raised when Dataset v5 cannot be assembled safely."""


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


def contains_phrase(text: str, term: str) -> bool:
    return term.lower() in text.lower()


def contains_role_code(text: str, role_code: str) -> bool:
    return re.search(rf"(?<![A-Za-zÀ-ÿ]){re.escape(role_code)}(?![A-Za-zÀ-ÿ])", text) is not None


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
    classic_role_codes = [str(term) for term in gates.get("classic_forbidden_role_codes", [])]
    broken_terms = [str(term) for term in gates.get("broken_terms", [])]

    for term in broken_terms:
        if term in text:
            raise AssemblyError(f"{prefix}: broken generated term {term!r}")

    if mode == "mantra":
        for term in mantra_terms:
            if contains_phrase(text, term):
                raise AssemblyError(f"{prefix}: forbidden Mantra term {term!r}")

    if mode == "classic":
        for term in classic_terms:
            if contains_phrase(text, term):
                raise AssemblyError(f"{prefix}: forbidden Classic term {term!r}")
        for role_code in classic_role_codes:
            if contains_role_code(text, role_code):
                raise AssemblyError(f"{prefix}: forbidden Classic role code {role_code!r}")


def validate_p4_record(
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

    required_tags = {"v5", "train", str(mode), str(task), focus_tag}
    missing_tags = sorted(required_tags - {str(tag) for tag in tags})
    if missing_tags:
        raise AssemblyError(f"{prefix}: missing tags: {', '.join(missing_tags)}")

    prompt = normalize_prompt(user_prompt(record))
    if prompt in forbidden_prompts:
        raise AssemblyError(f"{prefix}: P4 prompt matches a pagella prompt")

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


def validate_base_path(base_path: Path, quality_gates: dict[str, object]) -> None:
    normalized_base = str(base_path).replace("\\", "/")
    if bool(quality_gates.get("forbid_dataset_v3_as_base", False)) and "datasets/v3/" in normalized_base:
        raise AssemblyError("Dataset v5 must not use Dataset v3 as base")
    if bool(quality_gates.get("forbid_dataset_v4_as_base", False)) and "datasets/v4/" in normalized_base:
        raise AssemblyError("Dataset v5 must not use Dataset v4 as base")


def assemble_dataset(base_path: Path, manifest_path: Path, output_path: Path) -> dict[str, object]:
    manifest = load_yaml(manifest_path)
    quality_gates = manifest.get("quality_gates", {})
    validate_base_path(base_path, quality_gates)

    base_records = records_from_examples(base_path)
    eval_path = ROOT / manifest.get("blind_eval_path", "benchmarks/pagella_v0.jsonl")
    forbidden_prompts = load_pagella_prompts(
        eval_path,
        required=bool(quality_gates.get("forbid_eval_prompt_leakage", False)),
    )

    vocabulary_gates = manifest.get("mode_vocabulary_gates", {})
    min_quality = int(quality_gates.get("min_quality_score", 5))
    required_source = str(quality_gates.get("source", "v5_manual"))

    p4_records: list[dict[str, object]] = []
    for block in manifest.get("p4_blocks", []):
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
            validate_p4_record(
                record=record,
                path=block_path,
                index=index,
                min_quality=min_quality,
                required_source=required_source,
                focus_tag=focus_tag,
                forbidden_prompts=forbidden_prompts,
                vocabulary_gates=vocabulary_gates,
            )

        p4_records.extend(records)

    expected_total = int(manifest["p4_examples"])
    if len(p4_records) != expected_total:
        raise AssemblyError(f"expected {expected_total} P4 examples, got {len(p4_records)}")

    expected_p4_by_mode = manifest["p4_balance"]["by_mode"]
    p4_by_mode = count_modes(p4_records, expected_p4_by_mode)
    if p4_by_mode != expected_p4_by_mode:
        raise AssemblyError(f"expected P4 mode split {expected_p4_by_mode}, got {p4_by_mode}")

    output_records = [*base_records, *p4_records]
    validate_unique_prompts(output_records)

    expected_final_by_mode = manifest["final_balance"]["by_mode"]
    final_by_mode = count_modes(output_records, expected_final_by_mode)
    if final_by_mode != expected_final_by_mode:
        raise AssemblyError(f"expected final mode split {expected_final_by_mode}, got {final_by_mode}")

    write_jsonl(output_path, output_records)

    return {
        "base_examples": len(base_records),
        "p4_examples": len(p4_records),
        "total_examples": len(output_records),
        "p4_by_mode": p4_by_mode,
        "final_by_mode": final_by_mode,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble and validate Dataset v5.")
    parser.add_argument("--base", required=True, help="Final Dataset v2 JSONL path.")
    parser.add_argument("--manifest", required=True, help="Dataset v5 manifest path.")
    parser.add_argument("--output", required=True, help="Output Dataset v5 train JSONL path.")
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

    print("Dataset v5 assembled")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run targeted tests**

Run:

```bash
python -m pytest tests/test_assemble_dataset_v5.py -q
```

Expected: `7 passed`.

- [ ] **Step 3: Commit assembly script**

Run:

```bash
git add scripts/assemble_dataset_v5.py tests/test_assemble_dataset_v5.py
git commit -m "feat: add dataset v5 assembly validation"
```

Expected: commit succeeds.

---

### Task 4: Add Qwen v5 Training Config

**Files:**
- Create: `configs/sft/qwen25-3b-qlora-v5.yaml`
- Modify: `tests/test_training_configs.py`

- [ ] **Step 1: Add failing config test**

Append to `tests/test_training_configs.py`:

```python
def test_qwen25_lora_v5_config_points_to_dataset_v5() -> None:
    config_path = ROOT / "configs" / "sft" / "qwen25-3b-qlora-v5.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["data"]["train_path"] == "datasets/v5/train.jsonl"
    assert config["data"].get("eval_path") in (None, "")
    assert "pagella" not in config["data"]["train_path"]
    assert "datasets/v3" not in config["data"]["train_path"]
    assert "datasets/v4" not in config["data"]["train_path"]
    assert config["training"]["output_dir"] == "models/adapters/qwen25-3b-fantabrain-sft-v5"
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
python -m pytest tests/test_training_configs.py::test_qwen25_lora_v5_config_points_to_dataset_v5 -q
```

Expected: FAIL with `FileNotFoundError` for `qwen25-3b-qlora-v5.yaml`.

- [ ] **Step 3: Create v5 config**

Create `configs/sft/qwen25-3b-qlora-v5.yaml`:

```yaml
project:
  name: fantabrain-llm
  run_name: qwen25-3b-fantabrain-sft-v5

data:
  train_path: datasets/v5/train.jsonl
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
  output_dir: models/adapters/qwen25-3b-fantabrain-sft-v5
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

- [ ] **Step 4: Run config test**

Run:

```bash
python -m pytest tests/test_training_configs.py::test_qwen25_lora_v5_config_points_to_dataset_v5 -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit config**

Run:

```bash
git add configs/sft/qwen25-3b-qlora-v5.yaml tests/test_training_configs.py
git commit -m "feat: add qwen v5 training config"
```

Expected: commit succeeds.

---

### Task 5: Add v5 Runbook And README Commands

**Files:**
- Create: `docs/runbooks/qwen25-lora-v5.md`
- Modify: `README.md`

- [ ] **Step 1: Create runbook**

Create `docs/runbooks/qwen25-lora-v5.md` with the same structure as `docs/runbooks/qwen25-lora-v4.md`, replacing v4 names with v5 and using these key commands:

```bash
python scripts/assemble_dataset_v5.py \
  --base datasets/v2/train.jsonl \
  --manifest datasets/v5/manifest.yaml \
  --output datasets/v5/train.jsonl
```

Expected audit values:

```text
examples: 296
by mode: Counter({'mantra': 148, 'classic': 148})
p4 rows: 16
p4 by mode: Counter({'mantra': 8, 'classic': 8})
min quality: 4
```

Dataset zip command:

```bash
zip -r fantabrain-dataset-v5-296.zip datasets/v5 datasets/v2/train.jsonl datasets/v2/manifest.yaml
```

Training command:

```bash
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v5.yaml
```

Evaluation command:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v5 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v5-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

- [ ] **Step 2: Update README**

Add a compact Dataset v5 section to `README.md` near the existing dataset/runbook notes:

```markdown
### Dataset v5 / P4 micro

Dataset v5 is a 16-example micro corrective add-on over Dataset v2. It intentionally does not use Dataset v3 or Dataset v4 as base.

```bash
python scripts/assemble_dataset_v5.py \
  --base datasets/v2/train.jsonl \
  --manifest datasets/v5/manifest.yaml \
  --output datasets/v5/train.jsonl

python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v5.yaml
```

Use `docs/runbooks/qwen25-lora-v5.md` for the full Colab flow. Keep `qwen25-3b-fantabrain-sft-v2` as rollback unless v5 beats the promotion gates.
```
```

- [ ] **Step 3: Commit docs**

Run:

```bash
git add docs/runbooks/qwen25-lora-v5.md README.md
git commit -m "docs: add qwen v5 runbook"
```

Expected: commit succeeds.

---

### Task 6: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
python -m pytest tests/test_assemble_dataset_v5.py tests/test_training_configs.py -q
```

Expected: all targeted tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected: full suite passes.

- [ ] **Step 3: Inspect final branch state**

Run:

```bash
git status -sb
git log --oneline -6
```

Expected: clean working tree on `codex/p4-micro-dataset`, with commits for spec, plan, metadata, assembly validation, config, and docs.

- [ ] **Step 4: Push branch**

Run:

```bash
git push -u origin codex/p4-micro-dataset
```

Expected: branch pushed successfully.

---

## Manual Colab Authoring Checkpoint

After this implementation plan is complete, the user should author P4 rows manually in Colab or local files. The required draft paths are:

```text
datasets/v5/drafts/p4_block_001_mantra_no_modificatore.jsonl
datasets/v5/drafts/p4_block_002_classic_clean_vocab.jsonl
datasets/v5/drafts/p4_block_003_no_invented_modules.jsonl
datasets/v5/drafts/p4_block_004_italian_decision_clean.jsonl
```

Do not create those manual examples in the repo scaffold unless the user explicitly asks Codex to draft candidate rows for review. The user wants practice authoring dataset examples by hand.

---

## Self-Review

- Spec coverage: covered Dataset v5 shape, v2-only base, 16 examples, 8/8 mode balance, strict source/quality gates, token-aware role-code checks, Qwen v5 config, runbook, and promotion/rollback expectations.
- Placeholder scan: no unresolved planning markers. Every code-touching step includes concrete content or commands.
- Type consistency: v5 script names, config names, source `v5_manual`, tags `v5`, output adapter `qwen25-3b-fantabrain-sft-v5`, and dataset path `datasets/v5/train.jsonl` are consistent across tasks.
