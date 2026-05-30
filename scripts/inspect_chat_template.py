from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.dataset import load_examples, to_generation_messages  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect training vs eval chat-template rendering."
    )
    parser.add_argument("--model", required=True, help="Tokenizer model id or local path.")
    parser.add_argument("--eval", required=True, help="Evaluation JSONL path.")
    parser.add_argument("--case", type=int, default=1, help="1-based case number to inspect.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        print('Install training dependencies with: python -m pip install -e ".[train]"')
        return 1

    examples = load_examples(args.eval)
    if not 1 <= args.case <= len(examples):
        print(f"--case must be between 1 and {len(examples)}")
        return 1

    token = os.getenv("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    example = examples[args.case - 1]
    train_messages = [message.to_dict() for message in example.messages]
    eval_messages = [message.to_dict() for message in to_generation_messages(example)]

    train_rendered = tokenizer.apply_chat_template(
        train_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    eval_rendered = tokenizer.apply_chat_template(
        eval_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("== Tokenizer chat_template ==")
    print(tokenizer.chat_template)
    print()
    print("== Training render roles ==")
    print([message["role"] for message in train_messages])
    print()
    print("== Eval render roles ==")
    print([message["role"] for message in eval_messages])
    print()
    print("== Unified diff: training vs eval ==")
    print(
        "\n".join(
            difflib.unified_diff(
                train_rendered.splitlines(),
                eval_rendered.splitlines(),
                fromfile="training_full_example",
                tofile="eval_generation_prompt",
                lineterm="",
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
