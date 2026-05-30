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


def write_manifest(
    path: Path,
    block_path: Path,
    examples: int = 2,
    *,
    blind_eval_path: str = "benchmarks/pagella_v0.jsonl",
    mode_split: dict[str, int] | None = None,
    p2_balance: dict[str, int] | None = None,
) -> None:
    expected_split = mode_split or {"mantra": 1, "classic": 1}
    payload = {
        "version": "v2",
        "base_dataset": "base.jsonl",
        "train_path": "train.jsonl",
        "blind_eval_path": blind_eval_path,
        "p2_examples": examples,
        "p2_balance": {"by_mode": p2_balance or {"mantra": 1, "classic": 1}},
        "p2_blocks": [
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
                source="v1_manual",
                user="Base prompt?",
                assistant="Base answer.",
                tags=["v1", "train", "classic"],
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
                source="v2_manual",
                user="Prompt da non copiare?",
                assistant="Sceglierei la M.",
                tags=["v2", "train", "mantra", "lineup_advice", "test_focus"],
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
    write_manifest(manifest_path, block_path, blind_eval_path=str(pagella_path))

    with pytest.raises(AssemblyError, match="P2 prompt matches a pagella prompt"):
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
                mode="classic",
                task="rules_explanation",
                source="v2_manual",
                user="Prompt Classic 1?",
                assistant="Sceglierei il blocco difensivo.",
                tags=["v2", "train", "classic", "rules_explanation", "test_focus"],
            ),
            example(
                mode="classic",
                task="lineup_advice",
                source="v2_manual",
                user="Prompt Classic 2?",
                assistant="Sceglierei il voto sicuro.",
                tags=["v2", "train", "classic", "lineup_advice", "test_focus"],
            ),
        ],
    )
    write_manifest(
        manifest_path,
        block_path,
        mode_split={"mantra": 0, "classic": 2},
        p2_balance={"mantra": 0, "classic": 2},
    )

    summary = assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)

    assert summary["p2_by_mode"] == {"mantra": 0, "classic": 2}
