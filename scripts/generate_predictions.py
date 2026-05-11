from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.dataset import DatasetError, load_examples  # noqa: E402
from fantabrain_llm.inference import InferenceError, make_chat_client  # noqa: E402
from fantabrain_llm.predictions import build_predictions, write_prediction_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model predictions for a FantaBrain eval set.")
    parser.add_argument("--provider", required=True, choices=["echo", "transformers", "openai-compatible"])
    parser.add_argument("--model", required=True, help="Model id or served model name.")
    parser.add_argument("--eval", required=True, help="Evaluation JSONL path.")
    parser.add_argument("--run-name", required=True, help="Name for reports/runs/<run-name>.")
    parser.add_argument("--output-root", default="reports/runs", help="Report output directory.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        examples = load_examples(args.eval)
        client = make_chat_client(
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        responses: list[str] = []
        for index, example in enumerate(examples, start=1):
            print(f"Generating case {index}/{len(examples)}: {example.mode}/{example.task}")
            responses.append(client.generate(example.messages, mode=example.mode, task=example.task))

        predictions = build_predictions(
            examples=examples,
            responses=responses,
            provider=client.provider,
            model=client.model,
        )
        output_dir = write_prediction_run(
            predictions=predictions,
            run_name=args.run_name,
            eval_path=args.eval,
            output_root=args.output_root,
        )
    except (DatasetError, InferenceError) as exc:
        print(f"Prediction error: {exc}", file=sys.stderr)
        return 1

    print(f"Prediction run written to {output_dir}")
    print(f"  predictions: {output_dir / 'predictions.jsonl'}")
    print(f"  comparison:  {output_dir / 'comparison.md'}")
    print(f"  summary:     {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
