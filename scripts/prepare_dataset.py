from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.dataset import (  # noqa: E402
    DatasetError,
    filter_by_quality,
    load_examples,
    split_examples,
    to_sft_record,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and split FantaBrain JSONL data.")
    parser.add_argument("--input", required=True, help="Raw JSONL dataset path.")
    parser.add_argument("--output", required=True, help="Processed training JSONL path.")
    parser.add_argument("--eval-output", required=True, help="Processed eval JSONL path.")
    parser.add_argument("--eval-ratio", type=float, default=0.2, help="Fraction reserved for eval.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--min-quality", type=int, default=None, help="Optional minimum quality_score.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        examples = load_examples(args.input)
        filtered = filter_by_quality(examples, args.min_quality)
        if not filtered:
            raise DatasetError("No examples remain after filtering")

        train_examples, eval_examples = split_examples(filtered, args.eval_ratio, args.seed)
        train_count = write_jsonl(args.output, (to_sft_record(example) for example in train_examples))
        eval_count = write_jsonl(args.eval_output, (to_sft_record(example) for example in eval_examples))
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 1

    print("FantaBrain dataset prepared")
    print(f"  source: {args.input}")
    print(f"  train:  {args.output} ({train_count} examples)")
    print(f"  eval:   {args.eval_output} ({eval_count} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
