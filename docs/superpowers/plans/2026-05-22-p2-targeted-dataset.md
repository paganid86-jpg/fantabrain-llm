# P2 Targeted Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Dataset v2 as Dataset v1 plus 80 targeted P2 repair examples, then train and evaluate `qwen25-3b-fantabrain-sft-v2`.

**Architecture:** Keep authored data under `datasets/v2`, add a small assembly CLI that validates P2 drafts before writing `datasets/v2/train.jsonl`, and keep the Colab notebook as a runner rather than the source of truth. Training reuses the Qwen2.5-3B QLoRA path with explicit Colab T4 precision settings.

**Tech Stack:** Python 3.11, PyYAML, pytest, TRL/SFTTrainer, PEFT LoRA, Transformers, bitsandbytes on Colab T4.

---

## File Map

Create:

- `datasets/v2/README.md`: authoring contract for P2 and v2 assembly.
- `datasets/v2/manifest.yaml`: machine-readable P2 target counts and quality gates.
- `datasets/v2/drafts/.gitkeep`: keeps the drafts folder visible before manual JSONL blocks exist.
- `configs/sft/qwen25-3b-qlora-v2.yaml`: Qwen v2 training config.
- `scripts/assemble_dataset_v2.py`: validates P2 draft blocks and writes `datasets/v2/train.jsonl`.
- `tests/test_dataset_v2_manifest.py`: manifest balance and quality-gate tests.
- `tests/test_assemble_dataset_v2.py`: CLI/unit tests for v2 assembly behavior.
- `docs/runbooks/qwen25-lora-v2.md`: Colab cells for P2 authoring, v2 training, v2 pagella, and zipping outputs.

Modify:

- `scripts/train_lora.py`: pass optional precision and model-loading config keys that were needed on Colab T4.
- `tests/test_training_configs.py`: add v2 config expectations.
- `README.md`: add Dataset v2, v2 training, and v2 evaluation commands.

Do not commit:

- `datasets/v2/train.jsonl` until it has been validated and intentionally accepted.
- generated adapters under `models/adapters/`.
- generated reports under `reports/runs/`.
- downloaded zip artifacts.

---

### Task 1: Dataset v2 Manifest And Docs

**Files:**

- Create: `datasets/v2/manifest.yaml`
- Create: `datasets/v2/README.md`
- Create: `datasets/v2/drafts/.gitkeep`
- Create: `tests/test_dataset_v2_manifest.py`

- [ ] **Step 1: Write failing manifest tests**

Create `tests/test_dataset_v2_manifest.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_v2_manifest_keeps_p2_balanced() -> None:
    manifest_path = ROOT / "datasets" / "v2" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    assert manifest["version"] == "v2"
    assert manifest["base_dataset"] == "datasets/v1/train.jsonl"
    assert manifest["train_path"] == "datasets/v2/train.jsonl"
    assert manifest["p2_examples"] == 80
    assert manifest["p2_balance"]["by_mode"] == {"mantra": 40, "classic": 40}

    block_total = sum(block["examples"] for block in manifest["p2_blocks"])
    assert block_total == 80


def test_dataset_v2_manifest_tracks_repair_targets() -> None:
    manifest_path = ROOT / "datasets" / "v2" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    target_cases = set(manifest["repair_targets"]["primary_cases"])
    assert target_cases == {2, 3, 6, 9, 20, 27, 28, 34, 39}

    block_names = {block["name"] for block in manifest["p2_blocks"]}
    assert block_names == {
        "classic_modificatore",
        "mantra_role_codes_guardrail",
        "risk_varianza_decisioni",
        "refusal_grounded_clean",
        "italiano_asciutto_decision_first",
    }

    quality_gates = manifest["quality_gates"]
    assert quality_gates["min_quality_score"] == 5
    assert quality_gates["forbid_eval_prompt_leakage"] is True
    assert quality_gates["forbid_real_player_names"] is True
    assert quality_gates["forbid_specific_live_facts"] is True
    assert quality_gates["forbid_fake_rules"] is True
    assert quality_gates["keep_mantra_classic_same_level"] is True
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
python -m pytest tests/test_dataset_v2_manifest.py -q
```

Expected result:

```text
FAILED tests/test_dataset_v2_manifest.py::test_dataset_v2_manifest_keeps_p2_balanced
FAILED tests/test_dataset_v2_manifest.py::test_dataset_v2_manifest_tracks_repair_targets
```

The failure should be `FileNotFoundError` for `datasets/v2/manifest.yaml`.

- [ ] **Step 3: Create the v2 manifest**

Create `datasets/v2/manifest.yaml`:

```yaml
version: v2
status: draft
purpose: >
  Dataset v2 adds a targeted P2 repair set on top of Dataset v1. It focuses on
  the exact behavioral failures observed in the Qwen v1 pagella: Classic
  modificatore reasoning, Mantra role-code corruption, risk/varianza handling,
  grounded refusals, and cleaner decision-first Italian.

base_dataset: datasets/v1/train.jsonl
train_path: datasets/v2/train.jsonl
blind_eval_path: benchmarks/pagella_v0.jsonl

p2_examples: 80
p2_balance:
  by_mode:
    mantra: 40
    classic: 40

repair_targets:
  primary_cases:
    - 2
    - 3
    - 6
    - 9
    - 20
    - 27
    - 28
    - 34
    - 39
  secondary_cases:
    - 7
    - 15
    - 31
    - 35
    - 36

p2_blocks:
  - name: classic_modificatore
    path: datasets/v2/drafts/p2_block_001_classic_modificatore.jsonl
    examples: 20
    mode_split:
      mantra: 0
      classic: 20
    focus_tag: classic_modificatore
    target_cases: [6, 13, 15, 28, 34]
  - name: mantra_role_codes_guardrail
    path: datasets/v2/drafts/p2_block_002_mantra_role_codes_guardrail.jsonl
    examples: 20
    mode_split:
      mantra: 20
      classic: 0
    focus_tag: mantra_role_codes_guardrail
    target_cases: [2, 3, 9, 20, 27]
  - name: risk_varianza_decisioni
    path: datasets/v2/drafts/p2_block_003_risk_varianza_decisioni.jsonl
    examples: 16
    mode_split:
      mantra: 8
      classic: 8
    focus_tag: risk_varianza_decisioni
    target_cases: [31, 34, 35, 36]
  - name: refusal_grounded_clean
    path: datasets/v2/drafts/p2_block_004_refusal_grounded_clean.jsonl
    examples: 12
    mode_split:
      mantra: 6
      classic: 6
    focus_tag: refusal_grounded_clean
    target_cases: [37, 38, 39, 40]
  - name: italiano_asciutto_decision_first
    path: datasets/v2/drafts/p2_block_005_italiano_asciutto_decision_first.jsonl
    examples: 12
    mode_split:
      mantra: 6
      classic: 6
    focus_tag: italiano_asciutto_decision_first
    target_cases: [5, 8, 10, 12, 14, 16, 21, 22, 23, 24, 25, 30, 32, 33]

answer_contract:
  max_words: 110
  required_opening_markers:
    - Sceglierei
    - Preferirei
    - Eviterei
    - Non posso
  required_condition_markers:
    - solo se
    - a meno che
    - se invece
  max_missing_context_sentences: 1

quality_gates:
  min_quality_score: 5
  source: v2_manual
  no_pagella_training: true
  forbid_eval_prompt_leakage: true
  forbid_real_player_names: true
  forbid_specific_live_facts: true
  forbid_invented_percentages: true
  forbid_specific_scores_or_votes: true
  forbid_fake_rules: true
  forbid_role_code_corruption: true
  keep_mantra_classic_same_level: true
```

- [ ] **Step 4: Create the v2 README**

Create `datasets/v2/README.md`:

````markdown
# Dataset v2

Dataset v2 is the P2 targeted repair set for FantaBrain.

It is assembled as:

```text
datasets/v1/train.jsonl + datasets/v2/drafts/*.jsonl -> datasets/v2/train.jsonl
```

P2 adds 80 examples:

- 40 Mantra
- 40 Classic
- all `source: v2_manual`
- all `quality_score: 5`

## Blocks

1. `p2_block_001_classic_modificatore.jsonl` - Classic modificatore, portiere, voto medio, floor.
2. `p2_block_002_mantra_role_codes_guardrail.jsonl` - literal Mantra role codes and module legality.
3. `p2_block_003_risk_varianza_decisioni.jsonl` - favorite/underdog, floor/upside, doubtful starters.
4. `p2_block_004_refusal_grounded_clean.jsonl` - refusal without invented certainty.
5. `p2_block_005_italiano_asciutto_decision_first.jsonl` - concise decision-first style.

## Authoring Rules

- Do not copy prompts or expected answers from `benchmarks/pagella_v0.jsonl`.
- Do not use real player names.
- Do not invent percentages, votes, scores, prices, or rules.
- Mantra answers must use role codes literally: `E`, `M`, `T`, `W`, `Pc`, `Dc`, `A`.
- Classic answers must use Classic vocabulary: porta, difesa, centrocampo, attacco, voto, bonus, malus, modificatore.
- First sentence must contain the decision or refusal.
- Keep assistant answers under 110 words.

## Validation

Run:

```bash
python scripts/assemble_dataset_v2.py \
  --base datasets/v1/train.jsonl \
  --manifest datasets/v2/manifest.yaml \
  --output datasets/v2/train.jsonl
```

The command fails if a block is missing, counts are wrong, prompts are duplicated, quality is below 5, source is not `v2_manual`, or a P2 prompt exactly matches a pagella prompt.
````

- [ ] **Step 5: Keep the drafts directory visible**

Create `datasets/v2/drafts/.gitkeep` as an empty file.

- [ ] **Step 6: Run manifest tests**

Run:

```powershell
python -m pytest tests/test_dataset_v2_manifest.py -q
```

Expected result:

```text
2 passed
```

- [ ] **Step 7: Commit Task 1**

Run:

```powershell
git add datasets/v2/README.md datasets/v2/manifest.yaml datasets/v2/drafts/.gitkeep tests/test_dataset_v2_manifest.py
git commit -m "feat: scaffold dataset v2 targets"
```

---

### Task 2: Training Config And Colab Precision Guardrails

**Files:**

- Create: `configs/sft/qwen25-3b-qlora-v2.yaml`
- Modify: `tests/test_training_configs.py`
- Modify: `scripts/train_lora.py`

- [ ] **Step 1: Add the failing v2 config test**

Append this test to `tests/test_training_configs.py`:

```python
def test_qwen25_lora_v2_config_points_to_dataset_v2() -> None:
    config_path = ROOT / "configs" / "sft" / "qwen25-3b-qlora-v2.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["data"]["train_path"] == "datasets/v2/train.jsonl"
    assert config["data"].get("eval_path") in (None, "")
    assert "pagella" not in config["data"]["train_path"]
    assert config["training"]["output_dir"] == "models/adapters/qwen25-3b-fantabrain-sft-v2"
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
python -m pytest tests/test_training_configs.py::test_qwen25_lora_v2_config_points_to_dataset_v2 -q
```

Expected result:

```text
FAILED tests/test_training_configs.py::test_qwen25_lora_v2_config_points_to_dataset_v2
```

The failure should be `FileNotFoundError` for `configs/sft/qwen25-3b-qlora-v2.yaml`.

- [ ] **Step 3: Create the v2 training config**

Create `configs/sft/qwen25-3b-qlora-v2.yaml`:

```yaml
project:
  name: fantabrain-llm
  run_name: qwen25-3b-fantabrain-sft-v2

data:
  train_path: datasets/v2/train.jsonl
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
  output_dir: models/adapters/qwen25-3b-fantabrain-sft-v2
  max_length: 2048
  num_train_epochs: 2
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.0002
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

- [ ] **Step 4: Add precision and loading keys to `train_lora.py`**

Modify the import block in `scripts/train_lora.py`:

```python
import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any
```

In `build_model_init_kwargs`, after `kwargs` is created, add:

```python
    if "low_cpu_mem_usage" in model_config:
        kwargs["low_cpu_mem_usage"] = bool(model_config["low_cpu_mem_usage"])

    token = os.getenv("HF_TOKEN")
    if token:
        kwargs["token"] = token
```

In `main`, after `sft_values` is created and before the `eval_strategy` compatibility block, add:

```python
    for optional_key in (
        "bf16",
        "fp16",
        "gradient_checkpointing",
        "max_grad_norm",
        "optim",
    ):
        if optional_key in training_config:
            sft_values[optional_key] = training_config[optional_key]
```

- [ ] **Step 5: Run config tests**

Run:

```powershell
python -m pytest tests/test_training_configs.py -q
```

Expected result:

```text
3 passed
```

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add configs/sft/qwen25-3b-qlora-v2.yaml scripts/train_lora.py tests/test_training_configs.py
git commit -m "feat: add qwen v2 training config"
```

---

### Task 3: Dataset v2 Assembly CLI

**Files:**

- Create: `scripts/assemble_dataset_v2.py`
- Create: `tests/test_assemble_dataset_v2.py`

- [ ] **Step 1: Write assembly tests**

Create `tests/test_assemble_dataset_v2.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.assemble_dataset_v2 import AssemblyError, assemble_dataset


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


def write_manifest(path: Path, block_path: Path, examples: int = 2) -> None:
    payload = {
        "version": "v2",
        "base_dataset": "base.jsonl",
        "train_path": "train.jsonl",
        "blind_eval_path": "benchmarks/pagella_v0.jsonl",
        "p2_examples": examples,
        "p2_balance": {"by_mode": {"mantra": 1, "classic": 1}},
        "p2_blocks": [
            {
                "name": "test_block",
                "path": str(block_path),
                "examples": examples,
                "mode_split": {"mantra": 1, "classic": 1},
                "focus_tag": "test_focus",
                "target_cases": [2],
            }
        ],
        "quality_gates": {
            "min_quality_score": 5,
            "source": "v2_manual",
            "forbid_eval_prompt_leakage": True,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_assemble_dataset_appends_valid_p2_rows(tmp_path: Path) -> None:
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
                source="v1_manual",
                user="Base prompt?",
                assistant="Base answer.",
                tags=["v1", "train", "mantra"],
            )
        ],
    )
    write_jsonl(
        block_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v2_manual",
                user="P2 Mantra prompt?",
                assistant="Sceglierei la M solo se ti copre il modulo principale.",
                tags=["v2", "train", "mantra", "lineup_advice", "test_focus"],
            ),
            example(
                mode="classic",
                task="rules_explanation",
                source="v2_manual",
                user="P2 Classic prompt?",
                assistant="Sceglierei il blocco difensivo se alza il voto medio.",
                tags=["v2", "train", "classic", "rules_explanation", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path)

    summary = assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)

    assert summary == {
        "base_examples": 1,
        "p2_examples": 2,
        "total_examples": 3,
        "p2_by_mode": {"mantra": 1, "classic": 1},
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
                source="v1_manual",
                user="Prompt duplicato?",
                assistant="Base answer.",
                tags=["v1", "train", "mantra"],
            )
        ],
    )
    write_jsonl(
        block_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v2_manual",
                user="Prompt duplicato?",
                assistant="Sceglierei la M.",
                tags=["v2", "train", "mantra", "lineup_advice", "test_focus"],
            ),
            example(
                mode="classic",
                task="rules_explanation",
                source="v2_manual",
                user="Prompt diverso?",
                assistant="Sceglierei il blocco difensivo.",
                tags=["v2", "train", "classic", "rules_explanation", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="Duplicate user prompt"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_low_quality_p2_rows(tmp_path: Path) -> None:
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
                source="v1_manual",
                user="Base prompt?",
                assistant="Base answer.",
                tags=["v1", "train", "classic"],
            )
        ],
    )
    write_jsonl(
        block_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v2_manual",
                user="Prompt Mantra?",
                assistant="Sceglierei la M.",
                tags=["v2", "train", "mantra", "lineup_advice", "test_focus"],
                quality_score=4,
            ),
            example(
                mode="classic",
                task="rules_explanation",
                source="v2_manual",
                user="Prompt Classic?",
                assistant="Sceglierei il blocco difensivo.",
                tags=["v2", "train", "classic", "rules_explanation", "test_focus"],
            ),
        ],
    )
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="quality_score"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)
```

- [ ] **Step 2: Run the failing assembly tests**

Run:

```powershell
python -m pytest tests/test_assemble_dataset_v2.py -q
```

Expected result:

```text
ERROR tests/test_assemble_dataset_v2.py
```

The error should say `ModuleNotFoundError: No module named 'scripts.assemble_dataset_v2'`.

- [ ] **Step 3: Implement `scripts/assemble_dataset_v2.py`**

Create `scripts/assemble_dataset_v2.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.dataset import load_examples, to_sft_record, write_jsonl  # noqa: E402


class AssemblyError(ValueError):
    """Raised when Dataset v2 cannot be assembled safely."""


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


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def records_from_examples(path: Path) -> list[dict[str, object]]:
    return [to_sft_record(example) for example in load_examples(path)]


def load_pagella_prompts(eval_path: Path) -> set[str]:
    if not eval_path.exists():
        return set()
    return {normalize_prompt(user_prompt(record)) for record in records_from_examples(eval_path)}


def validate_p2_record(
    *,
    record: dict[str, object],
    path: Path,
    index: int,
    min_quality: int,
    required_source: str,
    focus_tag: str,
    forbidden_prompts: set[str],
) -> None:
    prefix = f"{path}:{index}"

    if record.get("source") != required_source:
        raise AssemblyError(f"{prefix}: source must be {required_source}")
    if record.get("quality_score") != min_quality:
        raise AssemblyError(f"{prefix}: quality_score must be {min_quality}")

    mode = record.get("mode")
    task = record.get("task")
    tags = record.get("tags")
    if mode not in {"mantra", "classic"}:
        raise AssemblyError(f"{prefix}: invalid mode {mode!r}")
    if not isinstance(task, str) or not task:
        raise AssemblyError(f"{prefix}: task is required")
    if not isinstance(tags, list):
        raise AssemblyError(f"{prefix}: tags must be a list")

    required_tags = {"v2", "train", str(mode), str(task), focus_tag}
    missing_tags = sorted(required_tags - {str(tag) for tag in tags})
    if missing_tags:
        raise AssemblyError(f"{prefix}: missing tags: {', '.join(missing_tags)}")

    prompt = normalize_prompt(user_prompt(record))
    if prompt in forbidden_prompts:
        raise AssemblyError(f"{prefix}: P2 prompt matches a pagella prompt")


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


def assemble_dataset(base_path: Path, manifest_path: Path, output_path: Path) -> dict[str, object]:
    manifest = load_yaml(manifest_path)
    base_records = records_from_examples(base_path)
    eval_path = ROOT / manifest.get("blind_eval_path", "benchmarks/pagella_v0.jsonl")
    forbidden_prompts = load_pagella_prompts(eval_path)

    quality_gates = manifest.get("quality_gates", {})
    min_quality = int(quality_gates.get("min_quality_score", 5))
    required_source = str(quality_gates.get("source", "v2_manual"))

    p2_records: list[dict[str, object]] = []
    for block in manifest.get("p2_blocks", []):
        block_path = resolve_manifest_path(manifest_path, str(block["path"]))
        expected_examples = int(block["examples"])
        expected_split = block["mode_split"]
        focus_tag = str(block["focus_tag"])

        records = records_from_examples(block_path)
        if len(records) != expected_examples:
            raise AssemblyError(
                f"{block_path}: expected {expected_examples} examples, got {len(records)}"
            )

        split = Counter(str(record["mode"]) for record in records)
        if dict(split) != expected_split:
            raise AssemblyError(f"{block_path}: expected mode split {expected_split}, got {dict(split)}")

        for index, record in enumerate(records, start=1):
            validate_p2_record(
                record=record,
                path=block_path,
                index=index,
                min_quality=min_quality,
                required_source=required_source,
                focus_tag=focus_tag,
                forbidden_prompts=forbidden_prompts,
            )

        p2_records.extend(records)

    expected_total = int(manifest["p2_examples"])
    if len(p2_records) != expected_total:
        raise AssemblyError(f"expected {expected_total} P2 examples, got {len(p2_records)}")

    expected_by_mode = manifest["p2_balance"]["by_mode"]
    p2_by_mode = dict(Counter(str(record["mode"]) for record in p2_records))
    if p2_by_mode != expected_by_mode:
        raise AssemblyError(f"expected P2 mode split {expected_by_mode}, got {p2_by_mode}")

    output_records = [*base_records, *p2_records]
    validate_unique_prompts(output_records)
    write_jsonl(output_path, output_records)

    return {
        "base_examples": len(base_records),
        "p2_examples": len(p2_records),
        "total_examples": len(output_records),
        "p2_by_mode": p2_by_mode,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble and validate Dataset v2.")
    parser.add_argument("--base", required=True, help="Final Dataset v1 JSONL path.")
    parser.add_argument("--manifest", required=True, help="Dataset v2 manifest path.")
    parser.add_argument("--output", required=True, help="Output Dataset v2 train JSONL path.")
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

    print("Dataset v2 assembled")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run assembly tests**

Run:

```powershell
python -m pytest tests/test_assemble_dataset_v2.py -q
```

Expected result:

```text
3 passed
```

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add scripts/assemble_dataset_v2.py tests/test_assemble_dataset_v2.py
git commit -m "feat: add dataset v2 assembly validation"
```

---

### Task 4: README And Runbook

**Files:**

- Create: `docs/runbooks/qwen25-lora-v2.md`
- Modify: `README.md`

- [ ] **Step 1: Add the v2 runbook**

Create `docs/runbooks/qwen25-lora-v2.md` with these sections:

````markdown
# Qwen2.5 3B LoRA v2 Runbook

This runbook trains `qwen25-3b-fantabrain-sft-v2` from Dataset v2 and evaluates it on Pagella v0.

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

## 2. Restore Dataset v1

```python
# Se datasets/v1/train.jsonl non esiste, carica lo zip/dataset v1 e ricostruiscilo prima di P2.
from pathlib import Path

%cd /content/fantabrain-llm
assert Path("datasets/v1/train.jsonl").exists(), "Ripristina datasets/v1/train.jsonl prima di assemblare v2"
print("v1 examples:", sum(1 for _ in open("datasets/v1/train.jsonl", encoding="utf-8")))
```

## 3. Author P2 Blocks

Create these files under `datasets/v2/drafts/`:

- `p2_block_001_classic_modificatore.jsonl`
- `p2_block_002_mantra_role_codes_guardrail.jsonl`
- `p2_block_003_risk_varianza_decisioni.jsonl`
- `p2_block_004_refusal_grounded_clean.jsonl`
- `p2_block_005_italiano_asciutto_decision_first.jsonl`

After each block:

```python
# Valida il blocco attraverso l'assemblatore completo.
%cd /content/fantabrain-llm
!python scripts/assemble_dataset_v2.py \
  --base datasets/v1/train.jsonl \
  --manifest datasets/v2/manifest.yaml \
  --output datasets/v2/train.jsonl
```

## 4. Train

```python
# Forgia v2.
%cd /content/fantabrain-llm
!python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v2.yaml
```

## 5. Evaluate

```python
# Pagella v2 con adapter.
%cd /content/fantabrain-llm
!python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v2 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v2-pagella-v0 \
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
!zip -r qwen25-3b-fantabrain-sft-v2-adapter.zip models/adapters/qwen25-3b-fantabrain-sft-v2
!zip -r qwen25-3b-fantabrain-sft-v2-pagella-v0.zip reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0

from google.colab import files
files.download("qwen25-3b-fantabrain-sft-v2-adapter.zip")
files.download("qwen25-3b-fantabrain-sft-v2-pagella-v0.zip")
```
````

- [ ] **Step 2: Update README**

In `README.md`, after the v1 forge command, add:

````markdown
Forgia Qwen v2, solo dopo aver completato `datasets/v2/train.jsonl`:

```bash
python -m pip install -e ".[train]"
python -m pip install -U "bitsandbytes>=0.46.1"
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v2.yaml
```

Il runbook operativo e `docs/runbooks/qwen25-lora-v2.md`.
````

After the adapter Qwen v0 pagella command, add the v2 equivalent:

````markdown
Pagella con adapter Qwen v2:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v2 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v2-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```
````

Also add this sentence near the Dataset v1 sentence:

```markdown
Dataset v2 e descritto in `datasets/v2/manifest.yaml` e nel runbook `docs/runbooks/qwen25-lora-v2.md`.
```

- [ ] **Step 3: Run docs-adjacent checks**

Run:

```powershell
python -m pytest tests/test_dataset_v2_manifest.py tests/test_training_configs.py tests/test_assemble_dataset_v2.py -q
```

Expected result:

```text
8 passed
```

- [ ] **Step 4: Commit Task 4**

Run:

```powershell
git add README.md docs/runbooks/qwen25-lora-v2.md
git commit -m "docs: add qwen v2 runbook"
```

---

### Task 5: Manual P2 Authoring Loop

**Files:**

- Create in Colab/runtime first: `datasets/v2/drafts/p2_block_001_classic_modificatore.jsonl`
- Create in Colab/runtime first: `datasets/v2/drafts/p2_block_002_mantra_role_codes_guardrail.jsonl`
- Create in Colab/runtime first: `datasets/v2/drafts/p2_block_003_risk_varianza_decisioni.jsonl`
- Create in Colab/runtime first: `datasets/v2/drafts/p2_block_004_refusal_grounded_clean.jsonl`
- Create in Colab/runtime first: `datasets/v2/drafts/p2_block_005_italiano_asciutto_decision_first.jsonl`
- Create after validation: `datasets/v2/train.jsonl`

- [ ] **Step 1: Restore final Dataset v1 in Colab**

Run:

```python
%cd /content/fantabrain-llm
from pathlib import Path

assert Path("datasets/v1/train.jsonl").exists(), "Manca datasets/v1/train.jsonl"
print("v1:", sum(1 for _ in open("datasets/v1/train.jsonl", encoding="utf-8")))
```

Expected result:

```text
v1: 200
```

If the result is not `200`, stop and restore the downloaded v1 dataset before continuing.

- [ ] **Step 2: Author Block 001**

Create `datasets/v2/drafts/p2_block_001_classic_modificatore.jsonl` with 20 Classic examples.

Every row must include:

- `mode: "classic"`
- `source: "v2_manual"`
- `quality_score: 5`
- tags including `v2`, `train`, `classic`, task name, `classic_modificatore`
- assistant answer under 110 words
- first sentence with `Sceglierei`, `Preferirei`, `Eviterei`, or `Non posso`

Validate:

```python
%cd /content/fantabrain-llm
!python scripts/assemble_dataset_v2.py \
  --base datasets/v1/train.jsonl \
  --manifest datasets/v2/manifest.yaml \
  --output datasets/v2/train.jsonl
```

Expected while only Block 001 exists:

```text
Assembly error: Dataset not found: datasets/v2/drafts/p2_block_002_mantra_role_codes_guardrail.jsonl
```

This is acceptable after Block 001 because the next block has not been written yet. If the error mentions Block 001 counts, mode split, source, quality, tags, duplicate prompt, or pagella leakage, fix Block 001 before continuing.

- [ ] **Step 3: Author Block 002**

Create `datasets/v2/drafts/p2_block_002_mantra_role_codes_guardrail.jsonl` with 20 Mantra examples.

Every row must include:

- `mode: "mantra"`
- `source: "v2_manual"`
- `quality_score: 5`
- tags including `v2`, `train`, `mantra`, task name, `mantra_role_codes_guardrail`
- literal use of at least one role code among `E`, `M`, `T`, `W`, `Pc`, `Dc`, `A`
- no `P` when the intended role is `Pc`
- no invented captain, budget threshold, percentage, or hidden rule

Validate with the same assembly command.

Expected while Blocks 001 and 002 exist:

```text
Assembly error: Dataset not found: datasets/v2/drafts/p2_block_003_risk_varianza_decisioni.jsonl
```

Fix any Block 001 or Block 002 validation error before continuing.

- [ ] **Step 4: Author Block 003**

Create `datasets/v2/drafts/p2_block_003_risk_varianza_decisioni.jsonl` with 16 examples:

- 8 Mantra
- 8 Classic
- focus tag: `risk_varianza_decisioni`

Every assistant answer must distinguish at least one of:

- favorito: protect floor first
- sfavorito: use targeted upside
- doubtful starter: avoid zero unless covered
- first-place state: stable repeatability beats spectacle

Validate with the same assembly command.

Expected while Blocks 001, 002, and 003 exist:

```text
Assembly error: Dataset not found: datasets/v2/drafts/p2_block_004_refusal_grounded_clean.jsonl
```

- [ ] **Step 5: Author Block 004**

Create `datasets/v2/drafts/p2_block_004_refusal_grounded_clean.jsonl` with 12 examples:

- 6 Mantra
- 6 Classic
- focus tag: `refusal_grounded_clean`

Every assistant answer must follow:

```text
Non posso inventare X. Posso pero stimare Y usando A, B e C. Mandami questi dati e ti restituisco una scelta ordinata.
```

The exact wording can change, but the three moves must remain:

- refuse impossible certainty
- offer a grounded estimate path
- ask for minimal data

Validate with the same assembly command.

Expected while Blocks 001, 002, 003, and 004 exist:

```text
Assembly error: Dataset not found: datasets/v2/drafts/p2_block_005_italiano_asciutto_decision_first.jsonl
```

- [ ] **Step 6: Author Block 005**

Create `datasets/v2/drafts/p2_block_005_italiano_asciutto_decision_first.jsonl` with 12 examples:

- 6 Mantra
- 6 Classic
- focus tag: `italiano_asciutto_decision_first`

Every assistant answer must be:

- 70 to 110 words
- decision-first
- one main tactical criterion
- one useful missing-data sentence at most
- no invented jargon

Validate with the same assembly command.

Expected result after all five blocks exist:

```text
Dataset v2 assembled
{
  "base_examples": 200,
  "p2_examples": 80,
  "total_examples": 280,
  "p2_by_mode": {
    "classic": 40,
    "mantra": 40
  }
}
```

The order of keys in `p2_by_mode` may differ.

- [ ] **Step 7: Final v2 dataset audit**

Run:

```python
%cd /content/fantabrain-llm
from collections import Counter
from fantabrain_llm.dataset import load_examples

examples = load_examples("datasets/v2/train.jsonl")
print("examples:", len(examples))
print("by mode:", Counter(example.mode for example in examples))
print("by task:", Counter(example.task for example in examples))
print("min quality:", min(example.quality_score or 0 for example in examples))
print("v2 rows:", sum(1 for example in examples if example.source == "v2_manual"))
```

Expected result:

```text
examples: 280
by mode: Counter({'mantra': 140, 'classic': 140})
min quality: 5
v2 rows: 80
```

The exact task distribution can vary if the block counts and mode split remain valid.

---

### Task 6: Train And Evaluate v2 In Colab

**Files:**

- Runtime output: `models/adapters/qwen25-3b-fantabrain-sft-v2`
- Runtime output: `reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0`

- [ ] **Step 1: Train v2**

Run in Colab:

```python
%cd /content/fantabrain-llm
!python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v2.yaml
```

Expected final line:

```text
Adapter saved to models/adapters/qwen25-3b-fantabrain-sft-v2
```

- [ ] **Step 2: Verify adapter files**

Run:

```python
%cd /content/fantabrain-llm
!find models/adapters/qwen25-3b-fantabrain-sft-v2 -maxdepth 1 -type f | sort
```

Expected files include:

```text
adapter_config.json
adapter_model.safetensors
```

- [ ] **Step 3: Run v2 pagella**

Run:

```python
%cd /content/fantabrain-llm
!python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v2 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v2-pagella-v0 \
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
Prediction run written to reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0
predictions: reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/predictions.jsonl
comparison:  reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/comparison.md
summary:     reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/summary.json
```

- [ ] **Step 4: Verify v2 pagella summary**

Run:

```python
%cd /content/fantabrain-llm
!head -n 40 reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/summary.json
!wc -l reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/predictions.jsonl
```

Expected fields:

```json
"examples": 40,
"adapter": "models/adapters/qwen25-3b-fantabrain-sft-v2",
"load_in_4bit": true,
"torch_dtype": "float16"
```

Expected row count:

```text
40 reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/predictions.jsonl
```

- [ ] **Step 5: Download artifacts**

Run:

```python
%cd /content/fantabrain-llm
!zip -r qwen25-3b-fantabrain-sft-v2-adapter.zip models/adapters/qwen25-3b-fantabrain-sft-v2
!zip -r qwen25-3b-fantabrain-sft-v2-pagella-v0.zip reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0

from google.colab import files
files.download("qwen25-3b-fantabrain-sft-v2-adapter.zip")
files.download("qwen25-3b-fantabrain-sft-v2-pagella-v0.zip")
```

---

### Task 7: Score v2 And Decide P3 Direction

**Files:**

- Runtime/local output: `scores_v2.template.csv`
- Runtime/local output: `scores_v2_codex_all.csv`
- Runtime/local output: `summary.scored.json`

- [ ] **Step 1: Create score template**

Run:

```bash
python scripts/create_scores_template.py \
  --predictions reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/predictions.jsonl \
  --output reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/scores_v2.template.csv
```

Expected result:

```text
Scores template written to reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/scores_v2.template.csv
```

- [ ] **Step 2: Manually score all 40 cases**

Use the same rubric as v1:

```csv
case,mode,tactical,grounded,clarity,tone,hallucination_free,notes
```

Scoring rule:

- `1` means harmful or badly wrong
- `3` means usable but weak or generic
- `5` means crisp FantaBrain-quality answer
- `hallucination_free=0` if it invents facts, roles, rules, prices, votes, percentages, or impossible certainty

- [ ] **Step 3: Generate scored summary**

Run:

```bash
python scripts/score_predictions.py \
  --predictions reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/predictions.jsonl \
  --scores reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/scores_v2_codex_all.csv
```

Expected result:

```text
Scored summary written to reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/summary.scored.json
Scores JSON written to reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/scores.json
```

- [ ] **Step 4: Compare v1 vs v2**

Compare:

- raw average
- effective average
- hallucination-free count
- capped cases
- Mantra average
- Classic average
- original P2 target cases: 2, 3, 6, 9, 20, 27, 28, 34, 39

Success target:

```text
effective_average >= 3.20
hallucination_free_count >= 36
capped_cases <= 4
```

- [ ] **Step 5: Commit repo scaffolding after local verification**

Run locally after Task 1 to Task 4 tests pass:

```powershell
python -m pytest tests/test_dataset_v2_manifest.py tests/test_training_configs.py tests/test_assemble_dataset_v2.py -q
python -m pytest -q
git status -sb
```

Expected result:

```text
8 passed
24+ passed
```

Then push the branch:

```powershell
git push origin codex/p1-dataset-v1
```

Update PR #4 with a short note:

```text
Adds P2 targeted dataset spec, Dataset v2 scaffold, v2 assembly validation, and Qwen v2 runbook. P2 examples and model artifacts remain out of git until manually validated.
```

---

## Self-Review Checklist

- The plan maps directly to `docs/superpowers/specs/2026-05-22-p2-targeted-dataset-design.md`.
- Dataset v2 remains balanced at 40 Mantra and 40 Classic P2 examples.
- P2 training does not use `benchmarks/pagella_v0.jsonl`.
- The assembly CLI rejects exact pagella prompt leakage.
- The train config points to `datasets/v2/train.jsonl`, not a pagella path.
- The evaluation command keeps the same decoding as v1.
- Generated adapters and reports remain outside git.
