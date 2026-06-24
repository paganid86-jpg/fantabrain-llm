from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.fallback_eval import (  # noqa: E402
    FallbackEvalError,
    run_fallback_eval,
    write_fallback_eval_outputs,
)
from fantabrain_llm.openai_fallback import (  # noqa: E402
    DEFAULT_FALLBACK_MODEL,
    OpenAIFallbackClient,
    OpenAIFallbackError,
)
from fantabrain_llm.output_filter import OutputFilterError  # noqa: E402
from fantabrain_llm.prediction_audit import (  # noqa: E402
    PredictionAuditError,
    load_prediction_records,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenAI fallback evaluation over FantaBrain predictions."
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for fallback reports.")
    parser.add_argument(
        "--fallback-model",
        default=DEFAULT_FALLBACK_MODEL,
        help="OpenAI model used only for blocked primary outputs.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=350,
        help="Max output tokens for fallback answers.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Fallback decoding temperature.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, client_factory=OpenAIFallbackClient) -> int:
    args = parse_args(argv)

    try:
        records = load_prediction_records(args.predictions)
        client = client_factory(
            model=args.fallback_model,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        )
        report = run_fallback_eval(records, fallback_client=client)
        json_path, markdown_path, predictions_path = write_fallback_eval_outputs(
            report,
            args.output_dir,
        )
    except (
        OSError,
        FallbackEvalError,
        OpenAIFallbackError,
        OutputFilterError,
        PredictionAuditError,
    ) as exc:
        print(f"Fallback eval error: {exc}", file=sys.stderr)
        return 1

    print(f"Fallback eval JSON written to {json_path}")
    print(f"Fallback eval Markdown written to {markdown_path}")
    print(f"Fallback predictions written to {predictions_path}")
    print(f"cases: {report.cases}")
    print(f"fallback_used_count: {report.fallback_used_count}")
    print(f"fallback_success_count: {report.fallback_success_count}")
    print(f"unresolved_safe_count: {report.unresolved_safe_count}")
    print(f"estimated_total_cost_usd: {report.estimated_total_cost_usd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
