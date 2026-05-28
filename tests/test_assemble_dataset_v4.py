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
                assistant="Sceglierei l'assetto gia coperto, usando solo i ruoli indicati.",
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

    write_jsonl(
        base_path,
        [
            example(
                mode="classic",
                task="lineup_advice",
                source="v2_manual",
                user="Base?",
                assistant="Base.",
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

    write_jsonl(
        base_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v2_manual",
                user="Base?",
                assistant="Base.",
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

    write_jsonl(
        base_path,
        [
            example(
                mode="classic",
                task="lineup_advice",
                source="v2_manual",
                user="Base?",
                assistant="Base.",
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
