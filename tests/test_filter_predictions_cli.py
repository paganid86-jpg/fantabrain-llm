from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prediction(case_id: int, prediction_text: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "mode": "mantra",
        "task": "lineup_advice",
        "tags": ["cli", "output-filter"],
        "prompt": "Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
        "expected": "Sceglierei 3-4-2-1.",
        "prediction": prediction_text,
        "provider": "echo",
        "model": "echo-baseline",
    }


def test_filter_predictions_cli_writes_outputs(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps(prediction(1, "Sceglierei 3-4-2-1.")),
                json.dumps(prediction(2, "Sceglierei 4-5-1.")),
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "filter"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/filter_predictions.py",
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Output filter JSON written to" in result.stdout
    assert "fallback: 1" in result.stdout
    assert (output_dir / "output_filter.json").exists()
    assert (output_dir / "output_filter.md").exists()


def test_filter_predictions_cli_reports_errors_cleanly(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(prediction(1, "Sceglierei 3-4-2-1.")),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output-file"
    output_dir.write_text("not a directory", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/filter_predictions.py",
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Output filter error:" in result.stderr
    assert "Traceback" not in result.stderr
