# P3 Cleanup Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Dataset v3 as Dataset v2 plus 40 P3 cleanup examples, then train and evaluate `qwen25-3b-fantabrain-sft-v3`.

**Architecture:** Keep authored P3 data under `datasets/v3`, mirror the existing v2 assembly pattern with a dedicated v3 manifest and assembly CLI, and keep Colab as the execution notebook rather than the source of truth. P3 does not broaden model knowledge; it adds a small anti-error cleanup layer for grounding, mode separation, and cleaner Italian.

**Tech Stack:** Python 3.11, PyYAML, pytest, TRL/SFTTrainer, PEFT LoRA, Transformers, bitsandbytes on Colab T4.

---

## File Map

Create:

- `datasets/v3/README.md`: authoring contract for P3 cleanup examples.
- `datasets/v3/manifest.yaml`: machine-readable P3 target counts, block paths, and quality gates.
- `datasets/v3/drafts/.gitkeep`: keeps the drafts folder visible before manual JSONL blocks exist.
- `scripts/assemble_dataset_v3.py`: validates P3 draft blocks and writes `datasets/v3/train.jsonl`.
- `tests/test_dataset_v3_manifest.py`: manifest balance and quality-gate tests.
- `tests/test_assemble_dataset_v3.py`: unit tests for v3 assembly behavior.
- `configs/sft/qwen25-3b-qlora-v3.yaml`: Qwen v3 training config.
- `docs/runbooks/qwen25-lora-v3.md`: Colab runbook for restore, authoring, training, pagella, and artifact download.

Modify:

- `tests/test_training_configs.py`: add v3 config expectations.
- `README.md`: add Dataset v3, v3 training, and v3 evaluation commands.

Do not commit:

- `datasets/v3/train.jsonl` until it has been manually authored, validated, and intentionally accepted.
- generated adapters under `models/adapters/`.
- generated reports under `reports/runs/`.
- downloaded zip artifacts.

---

### Task 1: Dataset v3 Manifest And Docs

**Files:**

- Create: `datasets/v3/manifest.yaml`
- Create: `datasets/v3/README.md`
- Create: `datasets/v3/drafts/.gitkeep`
- Create: `tests/test_dataset_v3_manifest.py`

- [ ] **Step 1: Write failing manifest tests**

Create `tests/test_dataset_v3_manifest.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_v3_manifest_keeps_p3_balanced() -> None:
    manifest_path = ROOT / "datasets" / "v3" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    assert manifest["version"] == "v3"
    assert manifest["base_dataset"] == "datasets/v2/train.jsonl"
    assert manifest["train_path"] == "datasets/v3/train.jsonl"
    assert manifest["p3_examples"] == 40
    assert manifest["p3_balance"]["by_mode"] == {"mantra": 20, "classic": 20}

    block_total = sum(block["examples"] for block in manifest["p3_blocks"])
    assert block_total == 40

    mode_totals = {
        "mantra": sum(block["mode_split"]["mantra"] for block in manifest["p3_blocks"]),
        "classic": sum(block["mode_split"]["classic"] for block in manifest["p3_blocks"]),
    }
    assert mode_totals == manifest["p3_balance"]["by_mode"]


def test_dataset_v3_manifest_tracks_cleanup_targets() -> None:
    manifest_path = ROOT / "datasets" / "v3" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    target_cases = set(manifest["repair_targets"]["primary_cases"])
    assert target_cases == {2, 5, 6, 7, 9, 10, 19, 25, 27, 28, 29, 32, 34, 37}
    assert set(manifest["repair_targets"]["secondary_cases"]) == {15, 31, 36, 40}

    block_names = {block["name"] for block in manifest["p3_blocks"]}
    assert block_names == {
        "mantra_no_module_invention",
        "classic_modificatore_clean",
        "refusal_stop_clean",
        "mantra_roles_no_cross_mode",
        "italiano_cleanup_decision_first",
    }

    quality_gates = manifest["quality_gates"]
    assert quality_gates["min_quality_score"] == 5
    assert quality_gates["source"] == "v3_manual"
    assert quality_gates["no_pagella_training"] is True
    assert quality_gates["forbid_eval_prompt_leakage"] is True
    assert quality_gates["forbid_real_player_names"] is True
    assert quality_gates["forbid_specific_live_facts"] is True
    assert quality_gates["forbid_invented_modules"] is True
    assert quality_gates["forbid_cross_mode_vocabulary"] is True
    assert quality_gates["forbid_broken_generated_words"] is True
    assert quality_gates["keep_mantra_classic_same_level"] is True
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
python -m pytest tests/test_dataset_v3_manifest.py -q
```

Expected result:

```text
FAILED tests/test_dataset_v3_manifest.py::test_dataset_v3_manifest_keeps_p3_balanced
FAILED tests/test_dataset_v3_manifest.py::test_dataset_v3_manifest_tracks_cleanup_targets
```

The failure should be `FileNotFoundError` for `datasets/v3/manifest.yaml`.

- [ ] **Step 3: Create the v3 manifest**

Create `datasets/v3/manifest.yaml`:

```yaml
version: v3
status: draft
purpose: >
  Dataset v3 adds a P3 cleanup set on top of Dataset v2. It focuses on
  anti-error behavior observed in the Qwen v2 pagella: no invented Mantra
  modules, no Classic/Mantra vocabulary leakage, cleaner Classic modificatore
  explanations, refusal answers that stop cleanly, and short decision-first
  Italian without malformed words.

base_dataset: datasets/v2/train.jsonl
train_path: datasets/v3/train.jsonl
blind_eval_path: benchmarks/pagella_v0.jsonl

p3_examples: 40
p3_balance:
  by_mode:
    mantra: 20
    classic: 20

repair_targets:
  primary_cases:
    - 2
    - 5
    - 6
    - 7
    - 9
    - 10
    - 19
    - 25
    - 27
    - 28
    - 29
    - 32
    - 34
    - 37
  secondary_cases:
    - 15
    - 31
    - 36
    - 40

p3_blocks:
  - name: mantra_no_module_invention
    path: datasets/v3/drafts/p3_block_001_mantra_no_module_invention.jsonl
    examples: 8
    mode_split:
      mantra: 8
      classic: 0
    focus_tag: mantra_no_module_invention
    target_cases: [2, 3, 31, 32]
  - name: classic_modificatore_clean
    path: datasets/v3/drafts/p3_block_002_classic_modificatore_clean.jsonl
    examples: 10
    mode_split:
      mantra: 0
      classic: 10
    focus_tag: classic_modificatore_clean
    target_cases: [6, 13, 15, 28, 29]
  - name: refusal_stop_clean
    path: datasets/v3/drafts/p3_block_003_refusal_stop_clean.jsonl
    examples: 8
    mode_split:
      mantra: 4
      classic: 4
    focus_tag: refusal_stop_clean
    target_cases: [37, 38, 39, 40]
  - name: mantra_roles_no_cross_mode
    path: datasets/v3/drafts/p3_block_004_mantra_roles_no_cross_mode.jsonl
    examples: 8
    mode_split:
      mantra: 8
      classic: 0
    focus_tag: mantra_roles_no_cross_mode
    target_cases: [9, 10, 19, 20, 25, 27]
  - name: italiano_cleanup_decision_first
    path: datasets/v3/drafts/p3_block_005_italiano_cleanup_decision_first.jsonl
    examples: 6
    mode_split:
      mantra: 0
      classic: 6
    focus_tag: italiano_cleanup_decision_first
    target_cases: [5, 7, 16, 22, 30, 36]

answer_contract:
  max_words: 95
  required_opening_markers:
    - Sceglierei
    - Preferirei
    - Eviterei
    - Non posso
  max_missing_context_sentences: 1
  max_conditional_branches: 1

quality_gates:
  min_quality_score: 5
  source: v3_manual
  no_pagella_training: true
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
```

- [ ] **Step 4: Create the v3 README**

Create `datasets/v3/README.md`:

````markdown
# Dataset v3

Dataset v3 is the P3 cleanup set for FantaBrain.

It is assembled as:

```text
datasets/v2/train.jsonl + datasets/v3/drafts/*.jsonl -> datasets/v3/train.jsonl
```

P3 adds 40 examples:

- 20 Mantra
- 20 Classic
- all `source: v3_manual`
- all `quality_score: 5`

## Blocks

1. `p3_block_001_mantra_no_module_invention.jsonl` - choose only among modules named by the user.
2. `p3_block_002_classic_modificatore_clean.jsonl` - clean Classic modifier explanations.
3. `p3_block_003_refusal_stop_clean.jsonl` - refusal that stops after grounded criteria and minimal data request.
4. `p3_block_004_mantra_roles_no_cross_mode.jsonl` - Mantra role-code reasoning without Classic leakage.
5. `p3_block_005_italiano_cleanup_decision_first.jsonl` - short, decision-first Italian.

## Authoring Rules

- Do not copy prompts or expected answers from `benchmarks/pagella_v0.jsonl`.
- Do not use real player names.
- Do not invent percentages, votes, scores, prices, modules, or rules.
- Mantra answers must use only modules named by the user, unless asking for missing module data.
- Classic answers must not use Mantra role-code logic.
- Refusal answers must not continue into invented specifics.
- First sentence must contain the decision or refusal.
- Keep assistant answers under 95 words.

## Validation

Run:

```bash
python scripts/assemble_dataset_v3.py \
  --base datasets/v2/train.jsonl \
  --manifest datasets/v3/manifest.yaml \
  --output datasets/v3/train.jsonl
```

The command fails if a block is missing, counts are wrong, prompts are duplicated, quality is below 5, source is not `v3_manual`, or a P3 prompt exactly matches a pagella prompt.
````

- [ ] **Step 5: Keep the drafts directory visible**

Create `datasets/v3/drafts/.gitkeep` as an empty file.

- [ ] **Step 6: Run manifest tests**

Run:

```powershell
python -m pytest tests/test_dataset_v3_manifest.py -q
```

Expected result:

```text
2 passed
```

- [ ] **Step 7: Commit Task 1**

Run:

```powershell
git add datasets/v3/README.md datasets/v3/manifest.yaml datasets/v3/drafts/.gitkeep tests/test_dataset_v3_manifest.py
git commit -m "feat: scaffold dataset v3 targets"
```

---

### Task 2: Dataset v3 Assembly CLI

**Files:**

- Create: `scripts/assemble_dataset_v3.py`
- Create: `tests/test_assemble_dataset_v3.py`

- [ ] **Step 1: Write assembly tests**

Create `tests/test_assemble_dataset_v3.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.assemble_dataset_v3 import AssemblyError, assemble_dataset


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
    examples: int = 2,
    *,
    blind_eval_path: str = "benchmarks/pagella_v0.jsonl",
    mode_split: dict[str, int] | None = None,
    p3_balance: dict[str, int] | None = None,
) -> None:
    expected_split = mode_split or {"mantra": 1, "classic": 1}
    payload = {
        "version": "v3",
        "base_dataset": "base.jsonl",
        "train_path": "train.jsonl",
        "blind_eval_path": blind_eval_path,
        "p3_examples": examples,
        "p3_balance": {"by_mode": p3_balance or {"mantra": 1, "classic": 1}},
        "p3_blocks": [
            {
                "name": "test_block",
                "path": str(block_path),
                "examples": examples,
                "mode_split": expected_split,
                "focus_tag": "test_focus",
                "target_cases": [2],
            }
        ],
        "quality_gates": {
            "min_quality_score": 5,
            "source": "v3_manual",
            "forbid_eval_prompt_leakage": True,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_assemble_dataset_appends_valid_p3_rows(tmp_path: Path) -> None:
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
                source="v3_manual",
                user="P3 Mantra prompt?",
                assistant="Sceglierei solo tra i moduli che mi hai dato.",
                tags=["v3", "train", "mantra", "lineup_advice", "test_focus"],
            ),
            example(
                mode="classic",
                task="rules_explanation",
                source="v3_manual",
                user="P3 Classic prompt?",
                assistant="Sceglierei il blocco difensivo se alza il voto medio.",
                tags=["v3", "train", "classic", "rules_explanation", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path)

    summary = assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)

    assert summary == {
        "base_examples": 1,
        "p3_examples": 2,
        "total_examples": 3,
        "p3_by_mode": {"mantra": 1, "classic": 1},
    }
    assert output_path.exists()
    assert sum(1 for _ in output_path.open("r", encoding="utf-8")) == 3


def test_assemble_dataset_rejects_duplicate_user_prompts(tmp_path: Path) -> None:
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
                user="Prompt duplicato?",
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
                source="v3_manual",
                user="Prompt duplicato?",
                assistant="Sceglierei solo tra i moduli indicati.",
                tags=["v3", "train", "mantra", "lineup_advice", "test_focus"],
            ),
            example(
                mode="classic",
                task="rules_explanation",
                source="v3_manual",
                user="Prompt diverso?",
                assistant="Sceglierei il blocco difensivo.",
                tags=["v3", "train", "classic", "rules_explanation", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="Duplicate user prompt"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_low_quality_p3_rows(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"

    write_jsonl(
        base_path,
        [
            example(
                mode="classic",
                task="lineup_advice",
                source="v2_manual",
                user="Base prompt?",
                assistant="Base answer.",
                tags=["v2", "train", "classic"],
            )
        ],
    )
    write_jsonl(
        block_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v3_manual",
                user="Prompt Mantra?",
                assistant="Sceglierei solo il modulo citato.",
                tags=["v3", "train", "mantra", "lineup_advice", "test_focus"],
                quality_score=4,
            ),
            example(
                mode="classic",
                task="rules_explanation",
                source="v3_manual",
                user="Prompt Classic?",
                assistant="Sceglierei il blocco difensivo.",
                tags=["v3", "train", "classic", "rules_explanation", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="quality_score"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_pagella_prompt_leakage(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    pagella_path = tmp_path / "pagella.jsonl"

    write_jsonl(
        base_path,
        [
            example(
                mode="classic",
                task="lineup_advice",
                source="v2_manual",
                user="Base prompt?",
                assistant="Base answer.",
                tags=["v2", "train", "classic"],
            )
        ],
    )
    write_jsonl(
        pagella_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="pagella",
                user="Prompt da non copiare?",
                assistant="Expected answer.",
                tags=["pagella"],
            )
        ],
    )
    write_jsonl(
        block_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v3_manual",
                user="Prompt da non copiare?",
                assistant="Sceglierei solo tra i moduli indicati.",
                tags=["v3", "train", "mantra", "lineup_advice", "test_focus"],
            ),
            example(
                mode="classic",
                task="rules_explanation",
                source="v3_manual",
                user="Prompt Classic?",
                assistant="Sceglierei il blocco difensivo.",
                tags=["v3", "train", "classic", "rules_explanation", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path, blind_eval_path=str(pagella_path))

    with pytest.raises(AssemblyError, match="P3 prompt matches a pagella prompt"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_requires_pagella_file_when_leakage_gate_is_enabled(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"

    write_jsonl(
        base_path,
        [
            example(
                mode="classic",
                task="lineup_advice",
                source="v2_manual",
                user="Base prompt?",
                assistant="Base answer.",
                tags=["v2", "train", "classic"],
            )
        ],
    )
    write_jsonl(
        block_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v3_manual",
                user="Prompt Mantra?",
                assistant="Sceglierei solo tra i moduli indicati.",
                tags=["v3", "train", "mantra", "lineup_advice", "test_focus"],
            ),
            example(
                mode="classic",
                task="rules_explanation",
                source="v3_manual",
                user="Prompt Classic?",
                assistant="Sceglierei il blocco difensivo.",
                tags=["v3", "train", "classic", "rules_explanation", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path, blind_eval_path=str(tmp_path / "missing.jsonl"))

    with pytest.raises(AssemblyError, match="Pagella eval file not found"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_accepts_zero_count_mode_split(tmp_path: Path) -> None:
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
                mode="classic",
                task="rules_explanation",
                source="v3_manual",
                user="Prompt Classic 1?",
                assistant="Sceglierei il blocco difensivo.",
                tags=["v3", "train", "classic", "rules_explanation", "test_focus"],
            ),
            example(
                mode="classic",
                task="lineup_advice",
                source="v3_manual",
                user="Prompt Classic 2?",
                assistant="Sceglierei il voto sicuro.",
                tags=["v3", "train", "classic", "lineup_advice", "test_focus"],
            ),
        ],
    )
    write_manifest(
        manifest_path,
        block_path,
        mode_split={"mantra": 0, "classic": 2},
        p3_balance={"mantra": 0, "classic": 2},
    )

    summary = assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)

    assert summary["p3_by_mode"] == {"mantra": 0, "classic": 2}
```

- [ ] **Step 2: Run the failing assembly tests**

Run:

```powershell
python -m pytest tests/test_assemble_dataset_v3.py -q
```

Expected result:

```text
FAILED tests/test_assemble_dataset_v3.py
```

The failure should be `ModuleNotFoundError` for `scripts.assemble_dataset_v3`.

- [ ] **Step 3: Create the v3 assembly script**

Create `scripts/assemble_dataset_v3.py` by copying the current structure of `scripts/assemble_dataset_v2.py`, then make these exact replacements throughout the new file:

```text
Dataset v2 -> Dataset v3
P2 -> P3
p2 -> p3
v2_manual -> v3_manual
p2_blocks -> p3_blocks
p2_examples -> p3_examples
p2_balance -> p3_balance
p2_records -> p3_records
validate_p2_record -> validate_p3_record
```

In the copied `validate_p3_record` function, make the pagella leakage error message:

```python
raise AssemblyError(f"{prefix}: P3 prompt matches a pagella prompt")
```

In `main`, make the success line:

```python
print("Dataset v3 assembled")
```

In the returned summary, use:

```python
return {
    "base_examples": len(base_records),
    "p3_examples": len(p3_records),
    "total_examples": len(output_records),
    "p3_by_mode": p3_by_mode,
}
```

- [ ] **Step 4: Run assembly tests**

Run:

```powershell
python -m pytest tests/test_assemble_dataset_v3.py -q
```

Expected result:

```text
6 passed
```

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add scripts/assemble_dataset_v3.py tests/test_assemble_dataset_v3.py
git commit -m "feat: add dataset v3 assembly validation"
```

---

### Task 3: Qwen v3 Training Config

**Files:**

- Create: `configs/sft/qwen25-3b-qlora-v3.yaml`
- Modify: `tests/test_training_configs.py`

- [ ] **Step 1: Add the failing v3 config test**

Append this test to `tests/test_training_configs.py`:

```python
def test_qwen25_lora_v3_config_points_to_dataset_v3() -> None:
    config_path = ROOT / "configs" / "sft" / "qwen25-3b-qlora-v3.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["data"]["train_path"] == "datasets/v3/train.jsonl"
    assert config["data"].get("eval_path") in (None, "")
    assert "pagella" not in config["data"]["train_path"]
    assert config["training"]["output_dir"] == "models/adapters/qwen25-3b-fantabrain-sft-v3"
    assert config["training"]["num_train_epochs"] == 2
    assert config["model"]["load_in_4bit"] is True
    assert config["model"]["torch_dtype"] == "float16"
    assert config["model"]["low_cpu_mem_usage"] is True
    assert config["training"]["bf16"] is False
    assert config["training"]["fp16"] is False
```

- [ ] **Step 2: Run the failing config test**

Run:

```powershell
python -m pytest tests/test_training_configs.py::test_qwen25_lora_v3_config_points_to_dataset_v3 -q
```

Expected result:

```text
FAILED tests/test_training_configs.py::test_qwen25_lora_v3_config_points_to_dataset_v3
```

The failure should be `FileNotFoundError` for `configs/sft/qwen25-3b-qlora-v3.yaml`.

- [ ] **Step 3: Create the v3 training config**

Create `configs/sft/qwen25-3b-qlora-v3.yaml`:

```yaml
project:
  name: fantabrain-llm
  run_name: qwen25-3b-fantabrain-sft-v3

data:
  train_path: datasets/v3/train.jsonl
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
  output_dir: models/adapters/qwen25-3b-fantabrain-sft-v3
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

```powershell
python -m pytest tests/test_training_configs.py -q
```

Expected result:

```text
4 passed
```

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add configs/sft/qwen25-3b-qlora-v3.yaml tests/test_training_configs.py
git commit -m "feat: add qwen v3 training config"
```

---

### Task 4: README And Runbook

**Files:**

- Create: `docs/runbooks/qwen25-lora-v3.md`
- Modify: `README.md`

- [ ] **Step 1: Add the v3 runbook**

Create `docs/runbooks/qwen25-lora-v3.md`:

````markdown
# Qwen2.5 3B LoRA v3 Runbook

This runbook trains `qwen25-3b-fantabrain-sft-v3` from Dataset v3 and evaluates it on Pagella v0.

## 1. Setup

```python
# Controlla GPU e CUDA.
!nvidia-smi

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

```python
# Clona o aggiorna la palestra.
import os
from pathlib import Path

repo = Path("/content/fantabrain-llm")
token = os.environ.get("GH_TOKEN")
assert token, "GH_TOKEN mancante nei Colab Secrets"

if not repo.exists():
    !git clone https://{token}@github.com/paganid86-jpg/fantabrain-llm.git /content/fantabrain-llm

%cd /content/fantabrain-llm
!git fetch origin codex/p1-dataset-v1
!git switch codex/p1-dataset-v1
!git pull --ff-only
```

```python
# Installa dipendenze repo e training.
%cd /content/fantabrain-llm
!python -m pip install -U pip
!python -m pip install -e ".[dev,train]"
!python -m pip install -U "bitsandbytes>=0.46.1"
```

## 2. Restore Dataset v2

```python
# Se datasets/v2/train.jsonl non esiste, carica fantabrain-dataset-v2-280.zip.
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

## 3. Author P3 Blocks

Create these files under `datasets/v3/drafts/`:

- `p3_block_001_mantra_no_module_invention.jsonl`
- `p3_block_002_classic_modificatore_clean.jsonl`
- `p3_block_003_refusal_stop_clean.jsonl`
- `p3_block_004_mantra_roles_no_cross_mode.jsonl`
- `p3_block_005_italiano_cleanup_decision_first.jsonl`

After each block:

```python
# Valida il blocco attraverso l'assemblatore completo.
%cd /content/fantabrain-llm
!python scripts/assemble_dataset_v3.py \
  --base datasets/v2/train.jsonl \
  --manifest datasets/v3/manifest.yaml \
  --output datasets/v3/train.jsonl
```

## 4. Train

```python
# Forgia v3.
%cd /content/fantabrain-llm
!python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v3.yaml
```

## 5. Evaluate

```python
# Pagella v3 con adapter.
%cd /content/fantabrain-llm
!python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v3 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v3-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

## 6. Download Artifacts

```python
# Scarica adapter e pagella.
%cd /content/fantabrain-llm
!zip -r qwen25-3b-fantabrain-sft-v3-adapter.zip models/adapters/qwen25-3b-fantabrain-sft-v3
!zip -r qwen25-3b-fantabrain-sft-v3-pagella-v0.zip reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0

from google.colab import files
files.download("qwen25-3b-fantabrain-sft-v3-adapter.zip")
files.download("qwen25-3b-fantabrain-sft-v3-pagella-v0.zip")
```
````

- [ ] **Step 2: Update README**

In `README.md`, add a Dataset v3 note near the existing Dataset v2 note:

```markdown
Dataset v3 e descritto in `datasets/v3/manifest.yaml` e nel runbook `docs/runbooks/qwen25-lora-v3.md`.
```

Add the v3 training command near the v2 command:

````markdown
Forgia Qwen v3, solo dopo aver completato `datasets/v3/train.jsonl`:

```bash
python -m pip install -e ".[train]"
python -m pip install -U "bitsandbytes>=0.46.1"
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v3.yaml
```
````

Add the v3 pagella command near the v2 pagella command:

````markdown
Pagella con adapter Qwen v3:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v3 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v3-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```
````

- [ ] **Step 3: Run docs-adjacent checks**

Run:

```powershell
python -m pytest tests/test_dataset_v3_manifest.py tests/test_training_configs.py tests/test_assemble_dataset_v3.py -q
```

Expected result:

```text
12 passed
```

- [ ] **Step 4: Commit Task 4**

Run:

```powershell
git add README.md docs/runbooks/qwen25-lora-v3.md
git commit -m "docs: add qwen v3 runbook"
```

---

### Task 5: Manual P3 Authoring Loop

**Files:**

- Create in Colab/runtime first: `datasets/v3/drafts/p3_block_001_mantra_no_module_invention.jsonl`
- Create in Colab/runtime first: `datasets/v3/drafts/p3_block_002_classic_modificatore_clean.jsonl`
- Create in Colab/runtime first: `datasets/v3/drafts/p3_block_003_refusal_stop_clean.jsonl`
- Create in Colab/runtime first: `datasets/v3/drafts/p3_block_004_mantra_roles_no_cross_mode.jsonl`
- Create in Colab/runtime first: `datasets/v3/drafts/p3_block_005_italiano_cleanup_decision_first.jsonl`
- Create after validation: `datasets/v3/train.jsonl`

- [ ] **Step 1: Restore final Dataset v2 in Colab**

Run:

```python
%cd /content/fantabrain-llm
from pathlib import Path

assert Path("datasets/v2/train.jsonl").exists(), "Manca datasets/v2/train.jsonl"
print("v2:", sum(1 for _ in open("datasets/v2/train.jsonl", encoding="utf-8")))
```

Expected result:

```text
v2: 280
```

If the result is not `280`, stop and restore `fantabrain-dataset-v2-280.zip`.

- [ ] **Step 2: Author Block 001**

Create `datasets/v3/drafts/p3_block_001_mantra_no_module_invention.jsonl` with 8 Mantra examples.

Every row must include:

- `mode: "mantra"`
- `source: "v3_manual"`
- `quality_score: 5`
- tags including `v3`, `train`, `mantra`, task name, `mantra_no_module_invention`
- assistant answer under 95 words
- first sentence with `Sceglierei`, `Preferirei`, `Eviterei`, or `Non posso`
- no module invented by the assistant

Validate:

```python
%cd /content/fantabrain-llm
!python scripts/assemble_dataset_v3.py \
  --base datasets/v2/train.jsonl \
  --manifest datasets/v3/manifest.yaml \
  --output datasets/v3/train.jsonl
```

Expected while only Block 001 exists:

```text
Assembly error: Dataset not found: datasets/v3/drafts/p3_block_002_classic_modificatore_clean.jsonl
```

This is acceptable after Block 001. If the error mentions Block 001 counts, mode split, source, quality, tags, duplicate prompt, or pagella leakage, fix Block 001 before continuing.

- [ ] **Step 3: Author Block 002**

Create `datasets/v3/drafts/p3_block_002_classic_modificatore_clean.jsonl` with 10 Classic examples.

Every row must include:

- `mode: "classic"`
- `source: "v3_manual"`
- `quality_score: 5`
- tags including `v3`, `train`, `classic`, task name, `classic_modificatore_clean`
- vocabulary centered on voto medio, difesa, portiere, floor, bonus, malus, modificatore
- no Mantra role codes and no rare-slot logic

Validate with the same assembly command.

Expected while Blocks 001 and 002 exist:

```text
Assembly error: Dataset not found: datasets/v3/drafts/p3_block_003_refusal_stop_clean.jsonl
```

- [ ] **Step 4: Author Block 003**

Create `datasets/v3/drafts/p3_block_003_refusal_stop_clean.jsonl` with 8 examples:

- 4 Mantra
- 4 Classic
- focus tag: `refusal_stop_clean`

Every assistant answer must follow this three-move structure:

```text
Non posso sapere X senza dati aggiornati o senza la tua rosa. Posso pero stimare Y usando A, B e C. Mandami questi dati e ti restituisco una scelta ordinata.
```

The exact wording can change, but it must not invent candidates, modules, rankings, vote projections, match facts, or role availability.

Validate with the same assembly command.

Expected while Blocks 001, 002, and 003 exist:

```text
Assembly error: Dataset not found: datasets/v3/drafts/p3_block_004_mantra_roles_no_cross_mode.jsonl
```

- [ ] **Step 5: Author Block 004**

Create `datasets/v3/drafts/p3_block_004_mantra_roles_no_cross_mode.jsonl` with 8 Mantra examples.

Every row must include:

- `mode: "mantra"`
- `source: "v3_manual"`
- `quality_score: 5`
- tags including `v3`, `train`, `mantra`, task name, `mantra_roles_no_cross_mode`
- at least one Mantra role code among `E`, `M`, `T`, `W`, `Pc`, `Dc`, `A`, `C`
- no Classic `modificatore difesa` unless the user explicitly mentions a hybrid rule
- a concrete explanation of which role absence breaks formations

Validate with the same assembly command.

Expected while Blocks 001, 002, 003, and 004 exist:

```text
Assembly error: Dataset not found: datasets/v3/drafts/p3_block_005_italiano_cleanup_decision_first.jsonl
```

- [ ] **Step 6: Author Block 005**

Create `datasets/v3/drafts/p3_block_005_italiano_cleanup_decision_first.jsonl` with 6 Classic examples.
- focus tag: `italiano_cleanup_decision_first`

Every assistant answer must be:

- 55 to 95 words
- decision-first
- one main tactical criterion
- one useful missing-data sentence at most
- free of malformed words such as `multiruoco`, `sicurata`, `offENSIVO`, `assettogliando`
- free of artificial jargon such as `codice`, `rotore`, and `punteggiatura`

Validate with the same assembly command.

Expected result after all five blocks exist:

```text
Dataset v3 assembled
{
  "base_examples": 280,
  "p3_examples": 40,
  "total_examples": 320,
  "p3_by_mode": {
    "mantra": 20,
    "classic": 20
  }
}
```

The order of keys in `p3_by_mode` may differ.

- [ ] **Step 7: Final v3 dataset audit**

Run:

```python
%cd /content/fantabrain-llm
import sys
from pathlib import Path
from collections import Counter

repo = Path("/content/fantabrain-llm")
if str(repo / "src") not in sys.path:
    sys.path.insert(0, str(repo / "src"))

from fantabrain_llm.dataset import load_examples

examples = load_examples("datasets/v3/train.jsonl")
print("examples:", len(examples))
print("by mode:", Counter(example.mode for example in examples))
print("by task:", Counter(example.task for example in examples))
print("min quality:", min(example.quality_score or 0 for example in examples))
print("v3 rows:", sum(1 for example in examples if example.source == "v3_manual"))
```

Expected result:

```text
examples: 320
by mode: Counter({'mantra': 160, 'classic': 160})
min quality: 4
v3 rows: 40
```

The `min quality` can remain `4` because older v0/v1 rows may contain quality 4. All P3 rows must be quality 5.

- [ ] **Step 8: Download Dataset v3 zip**

Run:

```python
%cd /content/fantabrain-llm
!zip -r fantabrain-dataset-v3-320.zip datasets/v3 datasets/v2/train.jsonl datasets/v2/manifest.yaml

from google.colab import files
files.download("fantabrain-dataset-v3-320.zip")
```

Expected local artifact:

```text
fantabrain-dataset-v3-320.zip
```

---

### Task 6: Train And Evaluate v3 In Colab

**Files:**

- Runtime output: `models/adapters/qwen25-3b-fantabrain-sft-v3`
- Runtime output: `reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0`

- [ ] **Step 1: Train v3**

Run in Colab:

```python
%cd /content/fantabrain-llm
!python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v3.yaml
```

Expected final line:

```text
Adapter saved to models/adapters/qwen25-3b-fantabrain-sft-v3
```

- [ ] **Step 2: Verify adapter files**

Run:

```python
%cd /content/fantabrain-llm
!find models/adapters/qwen25-3b-fantabrain-sft-v3 -maxdepth 1 -type f | sort
```

Expected files include:

```text
adapter_config.json
adapter_model.safetensors
```

- [ ] **Step 3: Run v3 pagella**

Run:

```python
%cd /content/fantabrain-llm
!python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v3 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v3-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

Expected final lines:

```text
Prediction run written to reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0
predictions: reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/predictions.jsonl
comparison:  reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/comparison.md
summary:     reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/summary.json
```

- [ ] **Step 4: Verify v3 pagella summary**

Run:

```python
%cd /content/fantabrain-llm
!head -n 40 reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/summary.json
!wc -l reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/predictions.jsonl
```

Expected fields:

```json
"examples": 40,
"adapter": "models/adapters/qwen25-3b-fantabrain-sft-v3",
"load_in_4bit": true,
"torch_dtype": "float16"
```

Expected row count:

```text
40 reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/predictions.jsonl
```

- [ ] **Step 5: Download artifacts**

Run:

```python
%cd /content/fantabrain-llm
!zip -r qwen25-3b-fantabrain-sft-v3-adapter.zip models/adapters/qwen25-3b-fantabrain-sft-v3
!zip -r qwen25-3b-fantabrain-sft-v3-pagella-v0.zip reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0

from google.colab import files
files.download("qwen25-3b-fantabrain-sft-v3-adapter.zip")
files.download("qwen25-3b-fantabrain-sft-v3-pagella-v0.zip")
```

---

### Task 7: Score v3 And Compare Against v2

**Files:**

- Runtime/local output: `scores_v3.template.csv`
- Runtime/local output: `scores_v3_codex_all.csv`
- Runtime/local output: `summary.scored.json`

- [ ] **Step 1: Create score template**

Run:

```bash
python scripts/create_scores_template.py \
  --predictions reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/predictions.jsonl \
  --output reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/scores_v3.template.csv
```

Expected result:

```text
Scores template written to reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/scores_v3.template.csv
```

- [ ] **Step 2: Manually score all 40 cases**

Use the same rubric as v2:

```csv
case,mode,tactical,grounded,clarity,tone,hallucination_free,notes
```

Scoring rule:

- `1` means harmful or badly wrong
- `3` means usable but weak or generic
- `5` means crisp FantaBrain-quality answer
- `hallucination_free=0` if it invents facts, roles, rules, prices, votes, percentages, modules, or impossible certainty

- [ ] **Step 3: Generate scored summary**

Run:

```bash
python scripts/score_predictions.py \
  --predictions reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/predictions.jsonl \
  --scores reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/scores_v3_codex_all.csv
```

Expected result:

```text
Scored summary written to reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/summary.scored.json
Scores JSON written to reports/runs/qwen25-3b-fantabrain-sft-v3-pagella-v0/scores.json
```

- [ ] **Step 4: Compare v2 vs v3**

Compare:

- raw average
- effective average
- hallucination-free count
- capped cases
- Mantra average
- Classic average
- invented module count in Mantra cases
- Classic/Mantra vocabulary leakage count
- original P3 target cases: 2, 5, 6, 7, 9, 10, 19, 25, 27, 28, 29, 32, 34, 37

Success target:

```text
hallucination_free_count >= 32
effective_average >= 2.90
raw_average >= 3.05
invented_mantra_modules == 0
classic_answers_using_mantra_role_logic == 0
```

- [ ] **Step 5: Push repo scaffolding after verification**

Run locally after Task 1 to Task 4 tests pass:

```powershell
python -m pytest tests/test_dataset_v3_manifest.py tests/test_training_configs.py tests/test_assemble_dataset_v3.py -q
python -m pytest -q
git status -sb
```

Expected result:

```text
12 passed
full pytest exits with code 0
```

Then push the branch:

```powershell
git push origin codex/p1-dataset-v1
```

Update PR #4 with a short note:

```text
Adds P3 cleanup dataset spec, Dataset v3 scaffold, v3 assembly validation, and Qwen v3 runbook. P3 examples and model artifacts remain out of git until manually validated.
```

---

## Self-Review Checklist

- The plan maps directly to `docs/superpowers/specs/2026-05-25-p3-cleanup-dataset-design.md`.
- Dataset v3 remains balanced at 20 Mantra and 20 Classic P3 examples.
- P3 training does not use `benchmarks/pagella_v0.jsonl`.
- The assembly CLI rejects exact pagella prompt leakage.
- The train config points to `datasets/v3/train.jsonl`, not a pagella path.
- The evaluation command keeps the same decoding as v2.
- Generated adapters and reports remain outside git.
- The P3 success target prioritizes hallucination-free improvement over broad raw-score chasing.
