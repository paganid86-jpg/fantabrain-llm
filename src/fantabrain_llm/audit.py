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
