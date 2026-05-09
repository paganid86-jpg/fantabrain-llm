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
        ExpectedMatrix(
            total=2,
            by_mode={"mantra": 1, "classic": 1},
            by_task_mode={
                ("lineup_advice", "mantra"): 1,
                ("lineup_advice", "classic"): 1,
            },
        ),
    )


def test_audit_examples_rejects_wrong_mode_count(tmp_path: Path) -> None:
    path = tmp_path / "mini.jsonl"
    write_jsonl(
        path,
        [
            row("mantra", "lineup_advice", "Prompt mantra 1"),
            row("mantra", "lineup_advice", "Prompt mantra 2"),
        ],
    )
    examples = load_examples(path)

    with pytest.raises(AuditError, match="mode mantra"):
        audit_examples(
            examples,
            ExpectedMatrix(
                total=2,
                by_mode={"mantra": 1, "classic": 1},
                by_task_mode={
                    ("lineup_advice", "mantra"): 1,
                    ("lineup_advice", "classic"): 1,
                },
            ),
        )


def test_audit_train_eval_split_rejects_duplicate_user_prompt(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(train_path, [row("mantra", "lineup_advice", "Stesso prompt")])
    write_jsonl(eval_path, [row("classic", "lineup_advice", "Stesso prompt")])

    with pytest.raises(AuditError, match="leakage"):
        audit_train_eval_split(load_examples(train_path), load_examples(eval_path))
