from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.dataset import DatasetError, load_examples, to_generation_messages  # noqa: E402
from fantabrain_llm.inference import InferenceError, make_chat_client  # noqa: E402
from fantabrain_llm.predictions import build_predictions, write_prediction_run  # noqa: E402
from fantabrain_llm.prompt_guards import (  # noqa: E402
    PromptGuardError,
    apply_prompt_guard,
    prompt_guard_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model predictions for a FantaBrain eval set."
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["echo", "transformers", "openai-compatible"],
    )
    parser.add_argument("--model", required=True, help="Model id or served model name.")
    parser.add_argument("--eval", required=True, help="Evaluation JSONL path.")
    parser.add_argument("--run-name", required=True, help="Name for reports/runs/<run-name>.")
    parser.add_argument("--output-root", default="reports/runs", help="Report output directory.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top_p.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalty for repeating tokens.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Block repeated n-grams of this size when supported by the provider.",
    )
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA adapter path.")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load transformers model in 4-bit.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", help="Torch dtype for transformers.")
    parser.add_argument(
        "--prompt-guard",
        default="none",
        choices=prompt_guard_names(),
        help="Optional inference-time prompt guard preset.",
    )
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
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            adapter=args.adapter,
            load_in_4bit=args.load_in_4bit,
            torch_dtype=args.torch_dtype,
        )

        responses: list[str] = []
        for index, example in enumerate(examples, start=1):
            print(f"Generating case {index}/{len(examples)}: {example.mode}/{example.task}")
            prompt_messages = apply_prompt_guard(
                to_generation_messages(example),
                mode=example.mode,
                preset=args.prompt_guard,
            )
            responses.append(
                client.generate(
                    prompt_messages,
                    mode=example.mode,
                    task=example.task,
                )
            )

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
            metadata={
                "adapter": args.adapter,
                "load_in_4bit": args.load_in_4bit,
                "torch_dtype": args.torch_dtype,
                "prompt_guard": args.prompt_guard,
                "decoding": {
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "no_repeat_ngram_size": args.no_repeat_ngram_size,
                },
            },
        )
    except (DatasetError, InferenceError, PromptGuardError) as exc:
        print(f"Prediction error: {exc}", file=sys.stderr)
        return 1

    print(f"Prediction run written to {output_dir}")
    print(f"  predictions: {output_dir / 'predictions.jsonl'}")
    print(f"  comparison:  {output_dir / 'comparison.md'}")
    print(f"  summary:     {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
