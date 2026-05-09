from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.audit import (  # noqa: E402
    DATASET_V0_MATRIX,
    PAGELLA_V0_MATRIX,
    AuditError,
    audit_examples,
    audit_train_eval_split,
)
from fantabrain_llm.dataset import DatasetError, load_examples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FantaBrain Dataset v0 and Pagella v0.")
    parser.add_argument("--train", default="datasets/v0/train.jsonl")
    parser.add_argument("--pagella", default="benchmarks/pagella_v0.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        train_examples = load_examples(args.train)
        pagella_examples = load_examples(args.pagella)
        audit_examples(train_examples, DATASET_V0_MATRIX)
        audit_examples(pagella_examples, PAGELLA_V0_MATRIX)
        audit_train_eval_split(train_examples, pagella_examples)
    except (DatasetError, AuditError) as exc:
        print(f"Audit error: {exc}", file=sys.stderr)
        return 1

    print("FantaBrain v0 audit passed")
    print(f"  train:   {args.train} ({len(train_examples)} examples)")
    print(f"  pagella: {args.pagella} ({len(pagella_examples)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
