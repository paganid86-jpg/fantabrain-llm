from __future__ import annotations

import json
from pathlib import Path

import pytest

from fantabrain_llm.dataset import DatasetError, load_examples, split_examples, to_sft_record
from fantabrain_llm.schema import TrainingExample


ROOT = Path(__file__).resolve().parents[1]


def test_seed_dataset_is_valid() -> None:
    examples = load_examples(ROOT / "examples" / "raw" / "seed_conversations.jsonl")

    assert len(examples) == 6
    assert {example.mode for example in examples} == {"mantra", "classic"}
    assert all(example.messages[0].role == "system" for example in examples)


def test_split_examples_is_deterministic() -> None:
    examples = load_examples(ROOT / "examples" / "raw" / "seed_conversations.jsonl")

    first_train, first_eval = split_examples(examples, eval_ratio=0.34, seed=7)
    second_train, second_eval = split_examples(examples, eval_ratio=0.34, seed=7)

    assert [example.task for example in first_train] == [example.task for example in second_train]
    assert [example.task for example in first_eval] == [example.task for example in second_eval]
    assert len(first_train) == 4
    assert len(first_eval) == 2


def test_invalid_dataset_reports_line_number(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text(
        json.dumps(
            {
                "mode": "mantra",
                "task": "lineup_advice",
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Question"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="invalid.jsonl:1"):
        load_examples(invalid_path)


def test_sft_record_preserves_chat_messages() -> None:
    example = TrainingExample.from_dict(
        {
            "mode": "classic",
            "task": "lineup_advice",
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
        }
    )

    record = to_sft_record(example)

    assert record["mode"] == "classic"
    assert record["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
