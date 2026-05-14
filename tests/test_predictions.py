from __future__ import annotations

import json
from pathlib import Path

from fantabrain_llm.dataset import load_examples
from fantabrain_llm.predictions import Prediction, build_predictions, write_prediction_run

ROOT = Path(__file__).resolve().parents[1]


def test_build_predictions_preserves_eval_context() -> None:
    examples = load_examples(ROOT / "examples" / "raw" / "seed_conversations.jsonl")[:1]

    predictions = build_predictions(
        examples=examples,
        responses=["Risposta baseline"],
        provider="echo",
        model="echo-baseline",
    )

    assert predictions == [
        Prediction(
            case_id=1,
            mode="mantra",
            task="lineup_advice",
            tags=["mantra", "lineup"],
            prompt=examples[0].messages[-2].content,
            expected=examples[0].messages[-1].content,
            prediction="Risposta baseline",
            provider="echo",
            model="echo-baseline",
        )
    ]


def test_write_prediction_run_creates_jsonl_markdown_and_summary(tmp_path: Path) -> None:
    prediction = Prediction(
        case_id=1,
        mode="classic",
        task="lineup_advice",
        tags=["classic", "lineup"],
        prompt="Domanda",
        expected="Risposta ideale",
        prediction="Risposta modello",
        provider="echo",
        model="echo-baseline",
    )

    output_dir = write_prediction_run(
        predictions=[prediction],
        run_name="echo-smoke",
        eval_path="benchmarks/pagella_v0.jsonl",
        output_root=tmp_path,
        metadata={"adapter": None, "decoding": {"temperature": 0.3}},
    )

    predictions_path = output_dir / "predictions.jsonl"
    comparison_path = output_dir / "comparison.md"
    summary_path = output_dir / "summary.json"

    assert predictions_path.exists()
    assert comparison_path.exists()
    assert summary_path.exists()

    record = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert record["prediction"] == "Risposta modello"
    assert record["expected"] == "Risposta ideale"

    comparison = comparison_path.read_text(encoding="utf-8")
    assert "## Case 1: classic / lineup_advice" in comparison
    assert "Risposta modello" in comparison
    assert "Risposta ideale" in comparison

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_name"] == "echo-smoke"
    assert summary["examples"] == 1
    assert summary["provider"] == "echo"
    assert summary["model"] == "echo-baseline"
    assert summary["adapter"] is None
    assert summary["decoding"]["temperature"] == 0.3
