from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

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
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = output_root / "echo-guarded"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert "Prediction run written" in result.stdout
    assert summary["prompt_guard"] == "mode_fence_v1"


class CapturingChatHandler(BaseHTTPRequestHandler):
    received_payload: dict[str, object] | None = None

    def do_POST(self) -> None:
        body_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(body_length).decode("utf-8")
        type(self).received_payload = json.loads(body)

        response = {"choices": [{"message": {"content": "Fake guarded response"}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        return


def test_generate_predictions_cli_sends_guarded_system_prompt_to_provider(
    tmp_path: Path,
) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
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
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "runs"
    CapturingChatHandler.received_payload = None
    server = HTTPServer(("127.0.0.1", 0), CapturingChatHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{server.server_port}"
        env["OPENAI_API_KEY"] = "test-key"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_predictions.py",
                "--provider",
                "openai-compatible",
                "--model",
                "fake-chat",
                "--eval",
                str(eval_path),
                "--run-name",
                "fake-guarded",
                "--output-root",
                str(output_root),
                "--prompt-guard",
                "mode_fence_v1",
            ],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    payload = CapturingChatHandler.received_payload
    assert payload is not None
    messages = payload["messages"]

    assert "Prediction run written" in result.stdout
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Prompt guard mode_fence_v1" in messages[0]["content"]
    assert "Regole Mantra" in messages[0]["content"]
    assert messages[1]["role"] == "user"
