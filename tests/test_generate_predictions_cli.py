from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generate_predictions_cli_writes_echo_run(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_predictions.py",
            "--provider",
            "echo",
            "--model",
            "echo-baseline",
            "--eval",
            "examples/raw/seed_conversations.jsonl",
            "--run-name",
            "echo-cli-smoke",
            "--output-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output_dir = tmp_path / "echo-cli-smoke"
    predictions_path = output_dir / "predictions.jsonl"
    comparison_path = output_dir / "comparison.md"
    summary_path = output_dir / "summary.json"

    assert predictions_path.exists()
    assert comparison_path.exists()
    assert summary_path.exists()

    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(predictions) == 6
    assert predictions[0]["provider"] == "echo"
    assert predictions[0]["model"] == "echo-baseline"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["examples"] == 6
    assert "Prediction run written" in result.stdout


def test_generate_predictions_cli_applies_prompt_guard_with_echo(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "mode": "mantra",
                        "task": "lineup_advice",
                        "source": "test",
                        "quality_score": 5,
                        "tags": ["test"],
                        "messages": [
                            {"role": "system", "content": "System"},
                            {"role": "user", "content": "Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?"},
                            {"role": "assistant", "content": "Gold"},
                        ],
                    }
                )
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_predictions.py",
            "--provider",
            "echo",
            "--model",
            "echo-baseline",
            "--eval",
            str(eval_path),
            "--run-name",
            "echo-guarded",
            "--output-root",
            str(output_root),
            "--prompt-guard",
            "mode_fence_v1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = output_root / "echo-guarded"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert "Prediction run written" in result.stdout
    assert summary["prompt_guard"] == "mode_fence_v1"
