from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from fantabrain_llm.openai_fallback import FallbackResponse, FallbackUsage

ROOT = Path(__file__).resolve().parents[1]


def prediction(case_id: int, prediction_text: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "mode": "mantra",
        "task": "lineup_advice",
        "tags": ["cli", "fallback"],
        "prompt": "Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
        "expected": "Sceglierei 3-4-2-1.",
        "prediction": prediction_text,
        "provider": "transformers",
        "model": "Qwen/Qwen2.5-3B-Instruct",
    }


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def generate(self, *, mode: str, task: str, prompt: str) -> FallbackResponse:
        self.calls.append({"mode": mode, "task": task, "prompt": prompt})
        return FallbackResponse(
            text="Sceglierei 3-4-2-1 perche resta nel prompt.",
            model="gpt-5.4-mini",
            usage=FallbackUsage(
                input_tokens=100,
                output_tokens=50,
                estimated_cost_usd=0.0003,
            ),
        )


def load_script_module():
    path = ROOT / "scripts" / "run_fallback_eval.py"
    spec = importlib.util.spec_from_file_location("run_fallback_eval", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_fallback_eval_cli_writes_outputs_with_mocked_client(
    tmp_path: Path,
    capsys,
) -> None:
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
    output_dir = tmp_path / "fallback"
    script = load_script_module()
    fake_client = FakeClient()

    def fake_factory(**kwargs):
        assert kwargs["model"] == "gpt-5.4-mini"
        assert kwargs["max_output_tokens"] == 350
        assert kwargs["temperature"] == 0.2
        return fake_client

    exit_code = script.main(
        [
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
        ],
        client_factory=fake_factory,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Fallback eval JSON written to" in captured.out
    assert "fallback_used_count: 1" in captured.out
    assert len(fake_client.calls) == 1
    assert (output_dir / "fallback_eval.json").exists()
    assert (output_dir / "fallback_eval.md").exists()
    assert (output_dir / "fallback_predictions.jsonl").exists()


def test_run_fallback_eval_cli_reports_errors_cleanly(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "fallback"
    script = load_script_module()

    exit_code = script.main(
        [
            "--predictions",
            str(tmp_path / "missing.jsonl"),
            "--output-dir",
            str(output_dir),
        ],
        client_factory=lambda **kwargs: FakeClient(),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Fallback eval error:" in captured.err
    assert "Traceback" not in captured.err
