from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from fantabrain_llm.schema import TrainingExample, ValidationError


class DatasetError(ValueError):
    """Raised when a JSONL dataset cannot be loaded or written."""


def load_examples(path: str | Path) -> list[TrainingExample]:
    source = Path(path)
    if not source.exists():
        raise DatasetError(f"Dataset not found: {source}")

    examples: list[TrainingExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc

            try:
                examples.append(TrainingExample.from_dict(payload))
            except ValidationError as exc:
                raise DatasetError(f"{source}:{line_number}: {exc}") from exc

    if not examples:
        raise DatasetError(f"Dataset has no examples: {source}")
    return examples


def filter_by_quality(
    examples: Iterable[TrainingExample],
    min_quality: int | None,
) -> list[TrainingExample]:
    if min_quality is None:
        return list(examples)
    if not 1 <= min_quality <= 5:
        raise DatasetError("min_quality must be between 1 and 5")
    return [
        example
        for example in examples
        if example.quality_score is not None and example.quality_score >= min_quality
    ]


def split_examples(
    examples: list[TrainingExample],
    eval_ratio: float,
    seed: int,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    if not 0 <= eval_ratio < 1:
        raise DatasetError("eval_ratio must be >= 0 and < 1")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    if eval_ratio == 0 or len(shuffled) == 1:
        return shuffled, []

    eval_count = max(1, round(len(shuffled) * eval_ratio))
    eval_count = min(eval_count, len(shuffled) - 1)
    eval_examples = shuffled[:eval_count]
    train_examples = shuffled[eval_count:]
    return train_examples, eval_examples


def to_sft_record(example: TrainingExample) -> dict[str, object]:
    return {
        "mode": example.mode,
        "task": example.task,
        "source": example.source,
        "quality_score": example.quality_score,
        "tags": example.tags,
        "messages": [message.to_dict() for message in example.messages],
    }


def write_jsonl(path: str | Path, records: Iterable[dict[str, object]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
