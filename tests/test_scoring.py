from __future__ import annotations

import json
from pathlib import Path

import pytest

from fantabrain_llm.predictions import Prediction, write_prediction_run
from fantabrain_llm.scoring import ScoreError, ScoreRow, aggregate_scores, load_scores_csv


def write_scores(path: Path, rows: list[str]) -> None:
    path.write_text(
        "case,mode,tactical,grounded,clarity,tone,hallucination_free,notes\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def test_score_row_caps_effective_average_when_hallucination_present() -> None:
    row = ScoreRow(
        case_id=1,
        mode=5,
        tactical=4,
        grounded=5,
        clarity=4,
        tone=5,
        hallucination_free=False,
        notes="Cita un dato non presente.",
    )

    assert row.raw_average == pytest.approx(4.6)
    assert row.effective_average == pytest.approx(1.0)


def test_aggregate_scores_reports_raw_effective_and_hallucination_rate() -> None:
    rows = [
        ScoreRow(
            case_id=1,
            mode=5,
            tactical=5,
            grounded=5,
            clarity=5,
            tone=5,
            hallucination_free=True,
            notes="",
        ),
        ScoreRow(
            case_id=2,
            mode=4,
            tactical=4,
            grounded=4,
            clarity=4,
            tone=4,
            hallucination_free=False,
            notes="Inventata soglia budget.",
        ),
    ]

    summary = aggregate_scores(rows)

    assert summary["cases"] == 2
    assert summary["raw_average"] == pytest.approx(4.5)
    assert summary["effective_average"] == pytest.approx(3.0)
    assert summary["hallucination_free_rate"] == pytest.approx(0.5)
    assert summary["capped_cases"] == 1


def test_load_scores_csv_validates_ranges_and_binary_hallucination(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.csv"
    write_scores(scores_path, ["1,5,4,5,4,5,1,ok", "2,4,4,3,4,4,0,allucinazione"])

    rows = load_scores_csv(scores_path)

    assert [row.case_id for row in rows] == [1, 2]
    assert rows[0].hallucination_free is True
    assert rows[1].hallucination_free is False

    write_scores(scores_path, ["1,6,4,5,4,5,1,fuori scala"])
    with pytest.raises(ScoreError, match="mode"):
        load_scores_csv(scores_path)

    write_scores(scores_path, ["1,5,4,5,4,5,2,non binario"])
    with pytest.raises(ScoreError, match="hallucination_free"):
        load_scores_csv(scores_path)


def test_score_predictions_cli_writes_scored_summary(tmp_path: Path) -> None:
    prediction = Prediction(
        case_id=1,
        mode="mantra",
        task="lineup_advice",
        tags=["mantra", "lineup"],
        prompt="Prompt",
        expected="Atteso",
        prediction="Risposta",
        provider="echo",
        model="echo-baseline",
    )
    output_dir = write_prediction_run(
        [prediction],
        run_name="echo-scored",
        eval_path="benchmarks/pagella_v0.jsonl",
        output_root=tmp_path,
    )
    scores_path = output_dir / "scores.csv"
    write_scores(scores_path, ["1,5,4,5,4,5,1,ok"])

    from scripts.score_predictions import main

    exit_code = main(
        [
            "--predictions",
            str(output_dir / "predictions.jsonl"),
            "--scores",
            str(scores_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "summary.scored.json").read_text(encoding="utf-8"))
    assert summary["run_name"] == "echo-scored"
    assert summary["scoring"]["effective_average"] == pytest.approx(4.6)
    assert (output_dir / "scores.json").exists()


def test_create_scores_template_cli_writes_blank_rows(tmp_path: Path) -> None:
    prediction = Prediction(
        case_id=1,
        mode="classic",
        task="trade_advice",
        tags=["classic", "trade"],
        prompt="Prompt",
        expected="Atteso",
        prediction="Risposta",
        provider="echo",
        model="echo-baseline",
    )
    output_dir = write_prediction_run(
        [prediction],
        run_name="echo-template",
        eval_path="benchmarks/pagella_v0.jsonl",
        output_root=tmp_path,
    )

    from scripts.create_scores_template import main

    exit_code = main(["--predictions", str(output_dir / "predictions.jsonl")])

    assert exit_code == 0
    template = (output_dir / "scores.template.csv").read_text(encoding="utf-8")
    assert template.splitlines()[0] == (
        "case,mode,tactical,grounded,clarity,tone,hallucination_free,notes"
    )
    assert template.splitlines()[1] == "1,,,,,,,"
