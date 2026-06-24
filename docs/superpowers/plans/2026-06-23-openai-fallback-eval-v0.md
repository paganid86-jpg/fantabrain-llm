# OpenAI Fallback Eval v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline fallback evaluation path that filters Qwen/LoRA predictions, calls `gpt-5.4-mini` only for blocked cases, filters fallback answers, and writes cost/quality reports.

**Architecture:** Keep the feature repo-local and eval-only. Add a small official OpenAI Responses API client, a deterministic fallback-eval core on top of `output_filter.py`, a report writer, and a CLI runner that consumes existing `predictions.jsonl` files.

**Tech Stack:** Python 3.11 stdlib, `urllib.request`, dataclasses, existing `fantabrain_llm.output_filter`, existing `fantabrain_llm.prediction_audit`, pytest, ruff.

---

## File Structure

- Create `src/fantabrain_llm/openai_fallback.py`: official OpenAI fallback client, payload builder, response parser, token usage, and cost estimator.
- Create `src/fantabrain_llm/fallback_eval.py`: fallback decision flow, final-source selection, report dataclasses, markdown/JSONL writers.
- Create `scripts/run_fallback_eval.py`: CLI wrapper for offline fallback eval over a prediction run.
- Create `tests/test_openai_fallback.py`: mocked Responses API tests; no real network.
- Create `tests/test_fallback_eval.py`: deterministic pipeline tests with a fake fallback client.
- Create `tests/test_run_fallback_eval_cli.py`: CLI tests using dependency injection; no real OpenAI call.
- Create `docs/runbooks/openai-fallback-eval-v0.md`: local/Colab runbook.
- Modify `README.md`: add the fallback eval command after output filter usage.

## Current Repo Contracts

Existing primary filter API:

```python
from fantabrain_llm.output_filter import (
    FilterAction,
    FilterDecision,
    filter_model_output,
)
```

Existing prediction loader:

```python
from fantabrain_llm.prediction_audit import load_prediction_records
```

Prediction records already contain:

```python
{
    "case_id": 1,
    "mode": "mantra",
    "task": "lineup_advice",
    "prompt": "Modalita Mantra...",
    "expected": "...",
    "prediction": "...",
    "provider": "transformers",
    "model": "Qwen/Qwen2.5-3B-Instruct",
}
```

Primary fallback trigger rule:

```python
primary_action in {FilterAction.FALLBACK, FilterAction.SAFE}
```

Fallback success rule:

```python
fallback_action in {FilterAction.PASS, FilterAction.PASS_WITH_WARNINGS}
```

Safe final answer:

```text
Damn, non ho abbastanza contesto per chiuderla con sicurezza.
Ti do una lettura prudente: evita scelte basate su dati non confermati e confronta titolarita, copertura, modalita e rischio prima di decidere.
```

---

### Task 1: Official OpenAI Fallback Client

**Files:**
- Create: `src/fantabrain_llm/openai_fallback.py`
- Test: `tests/test_openai_fallback.py`

- [ ] **Step 1: Write failing tests for API key, payload, parsing, and cost**

Create `tests/test_openai_fallback.py` with:

```python
from __future__ import annotations

import json
import urllib.error
from io import BytesIO

import pytest

from fantabrain_llm.openai_fallback import (
    FALLBACK_SYSTEM_INSTRUCTIONS,
    OpenAIFallbackClient,
    OpenAIFallbackError,
    estimate_cost_usd,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_missing_api_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAIFallbackError, match="OPENAI_API_KEY"):
        OpenAIFallbackClient()


def test_estimate_cost_for_gpt54_mini_standard_pricing() -> None:
    assert estimate_cost_usd(input_tokens=1000, output_tokens=500) == pytest.approx(
        0.003
    )


def test_generate_posts_responses_payload_and_extracts_output_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: int = 30):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse(
            {
                "model": "gpt-5.4-mini",
                "output_text": "Sceglierei la soluzione piu prudente.",
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                },
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAIFallbackClient(api_key="test-api-key", timeout_seconds=12)

    response = client.generate(
        mode="mantra",
        task="lineup_advice",
        prompt="Mantra: meglio 3-4-2-1 o 4-3-3?",
    )

    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["timeout"] == 12
    assert captured["headers"]["Authorization"] == "Bearer test-api-key"
    payload = captured["payload"]
    assert payload["model"] == "gpt-5.4-mini"
    assert payload["instructions"] == FALLBACK_SYSTEM_INSTRUCTIONS
    assert payload["max_output_tokens"] == 350
    assert payload["temperature"] == 0.2
    assert payload["tools"] == []
    assert payload["store"] is False
    assert payload["input"][0]["role"] == "user"
    assert "Modalita: mantra" in payload["input"][0]["content"]
    assert "Task: lineup_advice" in payload["input"][0]["content"]
    assert "3-4-2-1" in payload["input"][0]["content"]
    assert response.text == "Sceglierei la soluzione piu prudente."
    assert response.model == "gpt-5.4-mini"
    assert response.usage.input_tokens == 1000
    assert response.usage.output_tokens == 500
    assert response.usage.estimated_cost_usd == pytest.approx(0.003)


def test_generate_extracts_nested_output_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout: int = 30):
        return FakeHTTPResponse(
            {
                "model": "gpt-5.4-mini",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Eviterei la scelta piu fragile.",
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 8},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAIFallbackClient(api_key="test-api-key")

    response = client.generate(
        mode="classic",
        task="risk_management",
        prompt="Classic: due giocatori della stessa squadra?",
    )

    assert response.text == "Eviterei la scelta piu fragile."
    assert response.usage.estimated_cost_usd == pytest.approx(0.000045)


def test_provider_http_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout: int = 30):
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=BytesIO(b'{"error":{"message":"rate limited"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAIFallbackClient(api_key="test-api-key")

    with pytest.raises(OpenAIFallbackError, match="rate limited"):
        client.generate(mode="mantra", task="lineup_advice", prompt="Prompt")


def test_malformed_provider_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout: int = 30):
        return FakeHTTPResponse({"model": "gpt-5.4-mini", "output": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAIFallbackClient(api_key="test-api-key")

    with pytest.raises(OpenAIFallbackError, match="No text output"):
        client.generate(mode="classic", task="trade_advice", prompt="Prompt")
```

- [ ] **Step 2: Run client tests and confirm RED**

Run:

```bash
python -m pytest tests/test_openai_fallback.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'fantabrain_llm.openai_fallback'
```

- [ ] **Step 3: Implement the OpenAI fallback client**

Create `src/fantabrain_llm/openai_fallback.py`:

```python
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


GPT54_MINI_INPUT_USD_PER_MTOK = 0.75
GPT54_MINI_OUTPUT_USD_PER_MTOK = 4.50
DEFAULT_FALLBACK_MODEL = "gpt-5.4-mini"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

FALLBACK_SYSTEM_INSTRUCTIONS = """Sei il coach AI privato di FantaBrain.
Rispondi in italiano pulito, con decisione prima e spiegazione breve.
Rispetta la modalita richiesta: Mantra o Classic.
Non inventare moduli, nomi giocatori, voti futuri, probabilita o dati live non forniti.
Se manca contesto, dichiaralo e dai solo un criterio prudente."""


class OpenAIFallbackError(RuntimeError):
    """Raised when the official OpenAI fallback call cannot be completed."""


@dataclass(frozen=True)
class FallbackUsage:
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FallbackResponse:
    text: str
    model: str
    usage: FallbackUsage

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "model": self.model,
            "usage": self.usage.to_dict(),
        }


def estimate_cost_usd(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    model: str = DEFAULT_FALLBACK_MODEL,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    if model != DEFAULT_FALLBACK_MODEL:
        return None
    return (
        (input_tokens / 1_000_000) * GPT54_MINI_INPUT_USD_PER_MTOK
        + (output_tokens / 1_000_000) * GPT54_MINI_OUTPUT_USD_PER_MTOK
    )


class OpenAIFallbackClient:
    def __init__(
        self,
        *,
        model: str = DEFAULT_FALLBACK_MODEL,
        api_key: str | None = None,
        max_output_tokens: int = 350,
        temperature: float = 0.2,
        timeout_seconds: int = 30,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise OpenAIFallbackError(
                "OPENAI_API_KEY is required for fallback evaluation."
            )
        self.model = model
        self.api_key = resolved_key
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate(self, *, mode: str, task: str, prompt: str) -> FallbackResponse:
        payload = self._build_payload(mode=mode, task=task, prompt=prompt)
        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise OpenAIFallbackError(_format_http_error(exc)) from exc
        except urllib.error.URLError as exc:
            raise OpenAIFallbackError(f"OpenAI fallback request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise OpenAIFallbackError("OpenAI fallback returned invalid JSON.") from exc

        text = _extract_output_text(response_payload)
        usage = _extract_usage(response_payload, self.model)
        return FallbackResponse(
            text=text,
            model=str(response_payload.get("model") or self.model),
            usage=usage,
        )

    def _build_payload(self, *, mode: str, task: str, prompt: str) -> dict[str, object]:
        user_content = "\n".join(
            [
                f"Modalita: {mode}",
                f"Task: {task}",
                "",
                "Prompt utente:",
                prompt.strip(),
            ]
        )
        return {
            "model": self.model,
            "instructions": FALLBACK_SYSTEM_INSTRUCTIONS,
            "input": [{"role": "user", "content": user_content}],
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "tools": [],
            "store": False,
        }


def _extract_usage(payload: dict[str, Any], model: str) -> FallbackUsage:
    usage_payload = payload.get("usage")
    if not isinstance(usage_payload, dict):
        return FallbackUsage(input_tokens=None, output_tokens=None, estimated_cost_usd=None)

    input_tokens = _optional_int(usage_payload.get("input_tokens"))
    output_tokens = _optional_int(usage_payload.get("output_tokens"))
    return FallbackUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost_usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        ),
    )


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _extract_output_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)

    raise OpenAIFallbackError("No text output found in OpenAI fallback response.")


def _format_http_error(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = {}
    message = None
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        message = error["message"]
    detail = message or body or exc.reason
    return f"OpenAI fallback request failed with HTTP {exc.code}: {detail}"
```

- [ ] **Step 4: Run client tests and confirm GREEN**

Run:

```bash
python -m pytest tests/test_openai_fallback.py -q
```

Expected:

```text
6 passed
```

- [ ] **Step 5: Commit Task 1**

```bash
git add src/fantabrain_llm/openai_fallback.py tests/test_openai_fallback.py
git commit -m "feat: add openai fallback client"
```

---

### Task 2: Fallback Evaluation Core

**Files:**
- Create: `src/fantabrain_llm/fallback_eval.py`
- Test: `tests/test_fallback_eval.py`

- [ ] **Step 1: Write failing tests for fallback routing and final source**

Create `tests/test_fallback_eval.py` with:

```python
from __future__ import annotations

import pytest

from fantabrain_llm.fallback_eval import (
    SAFE_FALLBACK_RESPONSE,
    FinalSource,
    run_fallback_eval,
)
from fantabrain_llm.openai_fallback import FallbackResponse, FallbackUsage


def record(
    *,
    case_id: int,
    mode: str = "mantra",
    task: str = "lineup_advice",
    prompt: str = "Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
    prediction: str = "Sceglierei 3-4-2-1.",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "mode": mode,
        "task": task,
        "tags": ["test"],
        "prompt": prompt,
        "expected": "Expected answer.",
        "prediction": prediction,
        "provider": "transformers",
        "model": "Qwen/Qwen2.5-3B-Instruct",
    }


class FakeFallbackClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, str]] = []

    def generate(self, *, mode: str, task: str, prompt: str) -> FallbackResponse:
        self.calls.append({"mode": mode, "task": task, "prompt": prompt})
        text = self.responses.pop(0)
        return FallbackResponse(
            text=text,
            model="gpt-5.4-mini",
            usage=FallbackUsage(
                input_tokens=100,
                output_tokens=50,
                estimated_cost_usd=0.0003,
            ),
        )


def test_primary_pass_does_not_call_fallback() -> None:
    client = FakeFallbackClient(responses=[])

    report = run_fallback_eval(
        [record(case_id=1, prediction="Sceglierei 3-4-2-1.")],
        fallback_client=client,
    )

    assert client.calls == []
    result = report.results[0]
    assert result.final_source is FinalSource.PRIMARY
    assert result.final_prediction == "Sceglierei 3-4-2-1."
    assert report.fallback_used_count == 0


def test_primary_pass_with_warnings_does_not_call_fallback() -> None:
    client = FakeFallbackClient(responses=[])

    report = run_fallback_eval(
        [
            record(
                case_id=1,
                mode="classic",
                prompt="Modalita Classic. Chi schiero?",
                prediction="Sceglierei il titolare, ma evita scelta offENSIVO.",
            )
        ],
        fallback_client=client,
    )

    assert client.calls == []
    result = report.results[0]
    assert result.primary_action == "pass_with_warnings"
    assert result.final_source is FinalSource.PRIMARY


def test_primary_hard_failure_calls_fallback_and_uses_clean_fallback() -> None:
    client = FakeFallbackClient(
        responses=["Sceglierei 3-4-2-1 perche e gia citato e coperto."]
    )

    report = run_fallback_eval(
        [
            record(
                case_id=2,
                prediction="Sceglierei 4-5-1 per proteggere il centrocampo.",
            )
        ],
        fallback_client=client,
    )

    assert client.calls == [
        {
            "mode": "mantra",
            "task": "lineup_advice",
            "prompt": "Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
        }
    ]
    result = report.results[0]
    assert result.primary_action == "fallback"
    assert result.fallback_used is True
    assert result.fallback_action == "pass"
    assert result.final_source is FinalSource.FALLBACK
    assert result.final_prediction == "Sceglierei 3-4-2-1 perche e gia citato e coperto."
    assert report.fallback_success_count == 1


def test_primary_safe_empty_output_calls_fallback() -> None:
    client = FakeFallbackClient(responses=["Sceglierei la soluzione piu prudente."])

    report = run_fallback_eval(
        [record(case_id=3, prediction="  \n")],
        fallback_client=client,
    )

    assert len(client.calls) == 1
    result = report.results[0]
    assert result.primary_action == "safe"
    assert result.final_source is FinalSource.FALLBACK


def test_fallback_hard_failure_becomes_safe_final_answer() -> None:
    client = FakeFallbackClient(
        responses=["Sceglierei 4-5-1 anche se il prompt cita solo 3-4-2-1."]
    )

    report = run_fallback_eval(
        [
            record(
                case_id=4,
                prediction="Sceglierei 4-5-1 per proteggere il centrocampo.",
            )
        ],
        fallback_client=client,
    )

    result = report.results[0]
    assert result.fallback_action == "safe"
    assert result.final_source is FinalSource.SAFE
    assert result.final_prediction == SAFE_FALLBACK_RESPONSE
    assert report.unresolved_safe_count == 1


def test_report_counts_actions_sources_violations_and_cost() -> None:
    client = FakeFallbackClient(
        responses=["Sceglierei 3-4-2-1 perche resta nel prompt."]
    )

    report = run_fallback_eval(
        [
            record(case_id=1, prediction="Sceglierei 3-4-2-1."),
            record(case_id=2, prediction="Sceglierei 4-5-1."),
        ],
        fallback_client=client,
    )

    assert report.cases == 2
    assert report.primary_action_counts == {"pass": 1, "fallback": 1}
    assert report.final_source_counts == {"primary": 1, "fallback": 1}
    assert report.primary_violation_counts == {"invented_modules": 1}
    assert report.final_violation_counts == {}
    assert report.estimated_total_cost_usd == pytest.approx(0.0003)
    assert report.results[1].case_id == 2
    assert report.results[1].primary_violations[0]["case_id"] == 2
```

- [ ] **Step 2: Run core tests and confirm RED**

Run:

```bash
python -m pytest tests/test_fallback_eval.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'fantabrain_llm.fallback_eval'
```

- [ ] **Step 3: Implement fallback eval dataclasses and routing**

Create `src/fantabrain_llm/fallback_eval.py`:

```python
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from fantabrain_llm.output_filter import FilterAction, filter_model_output
from fantabrain_llm.openai_fallback import FallbackResponse, FallbackUsage


SAFE_FALLBACK_RESPONSE = (
    "Damn, non ho abbastanza contesto per chiuderla con sicurezza.\n"
    "Ti do una lettura prudente: evita scelte basate su dati non confermati "
    "e confronta titolarita, copertura, modalita e rischio prima di decidere."
)


class FallbackEvalError(ValueError):
    """Raised when fallback eval inputs are invalid."""


class FinalSource(StrEnum):
    PRIMARY = "primary"
    FALLBACK = "fallback"
    SAFE = "safe"


class FallbackClient(Protocol):
    def generate(self, *, mode: str, task: str, prompt: str) -> FallbackResponse: ...


@dataclass(frozen=True)
class FallbackCaseResult:
    case_id: int
    mode: str
    task: str
    prompt: str
    expected: str | None
    primary_prediction: str
    primary_action: str
    primary_reason: str
    primary_violations: list[dict[str, object]]
    fallback_used: bool
    fallback_prediction: str | None
    fallback_action: str | None
    fallback_reason: str | None
    fallback_violations: list[dict[str, object]]
    final_prediction: str
    final_source: FinalSource
    usage: FallbackUsage | None
    estimated_cost_usd: float | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["final_source"] = self.final_source.value
        payload["usage"] = self.usage.to_dict() if self.usage else None
        return payload


@dataclass(frozen=True)
class FallbackEvalReport:
    cases: int
    results: list[FallbackCaseResult]
    primary_action_counts: dict[str, int]
    fallback_used_count: int
    fallback_success_count: int
    final_source_counts: dict[str, int]
    unresolved_safe_count: int
    primary_violation_counts: dict[str, int]
    final_violation_counts: dict[str, int]
    estimated_total_cost_usd: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "cases": self.cases,
            "primary_action_counts": self.primary_action_counts,
            "fallback_used_count": self.fallback_used_count,
            "fallback_success_count": self.fallback_success_count,
            "final_source_counts": self.final_source_counts,
            "unresolved_safe_count": self.unresolved_safe_count,
            "primary_violation_counts": self.primary_violation_counts,
            "final_violation_counts": self.final_violation_counts,
            "estimated_total_cost_usd": self.estimated_total_cost_usd,
            "results": [result.to_dict() for result in self.results],
        }


def run_fallback_eval(
    records: list[dict[str, object]],
    *,
    fallback_client: FallbackClient,
) -> FallbackEvalReport:
    results: list[FallbackCaseResult] = []

    for record in records:
        case_id = _require_int(record, "case_id")
        mode = _require_str(record, "mode")
        task = _require_str(record, "task")
        prompt = _require_str(record, "prompt")
        prediction = _require_prediction(record)
        expected = record.get("expected")
        if expected is not None and not isinstance(expected, str):
            raise FallbackEvalError(f"case {case_id}: expected must be a string")

        primary_decision = filter_model_output(
            mode=mode,
            task=task,
            prompt=prompt,
            prediction=prediction,
            case_id=case_id,
        )

        if primary_decision.action in {FilterAction.PASS, FilterAction.PASS_WITH_WARNINGS}:
            results.append(
                FallbackCaseResult(
                    case_id=case_id,
                    mode=mode,
                    task=task,
                    prompt=prompt,
                    expected=expected,
                    primary_prediction=prediction,
                    primary_action=primary_decision.action.value,
                    primary_reason=primary_decision.reason,
                    primary_violations=[
                        violation.to_dict() for violation in primary_decision.violations
                    ],
                    fallback_used=False,
                    fallback_prediction=None,
                    fallback_action=None,
                    fallback_reason=None,
                    fallback_violations=[],
                    final_prediction=prediction,
                    final_source=FinalSource.PRIMARY,
                    usage=None,
                    estimated_cost_usd=None,
                )
            )
            continue

        fallback_response = fallback_client.generate(mode=mode, task=task, prompt=prompt)
        fallback_decision = filter_model_output(
            mode=mode,
            task=task,
            prompt=prompt,
            prediction=fallback_response.text,
            fallback_failed=True,
            case_id=case_id,
        )

        fallback_success = fallback_decision.action in {
            FilterAction.PASS,
            FilterAction.PASS_WITH_WARNINGS,
        }
        final_source = FinalSource.FALLBACK if fallback_success else FinalSource.SAFE
        final_prediction = fallback_response.text if fallback_success else SAFE_FALLBACK_RESPONSE

        results.append(
            FallbackCaseResult(
                case_id=case_id,
                mode=mode,
                task=task,
                prompt=prompt,
                expected=expected,
                primary_prediction=prediction,
                primary_action=primary_decision.action.value,
                primary_reason=primary_decision.reason,
                primary_violations=[
                    violation.to_dict() for violation in primary_decision.violations
                ],
                fallback_used=True,
                fallback_prediction=fallback_response.text,
                fallback_action=fallback_decision.action.value,
                fallback_reason=fallback_decision.reason,
                fallback_violations=[
                    violation.to_dict() for violation in fallback_decision.violations
                ],
                final_prediction=final_prediction,
                final_source=final_source,
                usage=fallback_response.usage,
                estimated_cost_usd=fallback_response.usage.estimated_cost_usd,
            )
        )

    return _build_report(results)


def _build_report(results: list[FallbackCaseResult]) -> FallbackEvalReport:
    primary_action_counts = Counter(result.primary_action for result in results)
    fallback_success_count = sum(
        1 for result in results if result.final_source is FinalSource.FALLBACK
    )
    final_source_counts = Counter(result.final_source.value for result in results)
    unresolved_safe_count = final_source_counts.get(FinalSource.SAFE.value, 0)
    primary_violation_counts = Counter(
        str(violation["check"])
        for result in results
        for violation in result.primary_violations
    )
    final_violation_counts = Counter(
        str(violation["check"])
        for result in results
        for violation in _final_violations(result)
    )
    costs = [
        result.estimated_cost_usd
        for result in results
        if result.estimated_cost_usd is not None
    ]
    total_cost = sum(costs) if costs else None
    return FallbackEvalReport(
        cases=len(results),
        results=results,
        primary_action_counts=dict(primary_action_counts),
        fallback_used_count=sum(1 for result in results if result.fallback_used),
        fallback_success_count=fallback_success_count,
        final_source_counts=dict(final_source_counts),
        unresolved_safe_count=unresolved_safe_count,
        primary_violation_counts=dict(primary_violation_counts),
        final_violation_counts=dict(final_violation_counts),
        estimated_total_cost_usd=total_cost,
    )


def _final_violations(result: FallbackCaseResult) -> list[dict[str, object]]:
    if result.final_source is FinalSource.PRIMARY:
        return result.primary_violations
    if result.final_source is FinalSource.FALLBACK:
        return result.fallback_violations
    return result.fallback_violations


def _require_str(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise FallbackEvalError(f"prediction record {field!r} must be a non-empty string")
    return value.strip()


def _require_prediction(record: dict[str, object]) -> str:
    value = record.get("prediction")
    if not isinstance(value, str):
        raise FallbackEvalError("prediction record 'prediction' must be a string")
    return value


def _require_int(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FallbackEvalError(f"prediction record {field!r} must be an integer")
    return value
```

- [ ] **Step 4: Run core tests and confirm GREEN**

Run:

```bash
python -m pytest tests/test_fallback_eval.py -q
```

Expected:

```text
6 passed
```

- [ ] **Step 5: Commit Task 2**

```bash
git add src/fantabrain_llm/fallback_eval.py tests/test_fallback_eval.py
git commit -m "feat: add fallback eval core"
```

---

### Task 3: Fallback Eval Output Writers

**Files:**
- Modify: `src/fantabrain_llm/fallback_eval.py`
- Modify: `tests/test_fallback_eval.py`

- [ ] **Step 1: Add failing writer tests**

Append to `tests/test_fallback_eval.py`:

```python
import json
from pathlib import Path

from fantabrain_llm.fallback_eval import (
    render_fallback_eval_markdown,
    write_fallback_eval_outputs,
)


def test_render_fallback_eval_markdown_includes_summary_and_flagged_case() -> None:
    client = FakeFallbackClient(
        responses=["Sceglierei 3-4-2-1 perche resta nel prompt."]
    )
    report = run_fallback_eval(
        [
            record(case_id=1, prediction="Sceglierei 3-4-2-1."),
            record(case_id=2, prediction="Sceglierei 4-5-1."),
        ],
        fallback_client=client,
    )

    markdown = render_fallback_eval_markdown(report)

    assert "# OpenAI Fallback Eval Report" in markdown
    assert "Cases: 2" in markdown
    assert "fallback_used_count: 1" in markdown
    assert "Case 2: mantra / lineup_advice" in markdown
    assert "Final source: fallback" in markdown


def test_write_fallback_eval_outputs_writes_json_markdown_and_jsonl(
    tmp_path: Path,
) -> None:
    client = FakeFallbackClient(
        responses=["Sceglierei 3-4-2-1 perche resta nel prompt."]
    )
    report = run_fallback_eval(
        [
            record(case_id=1, prediction="Sceglierei 3-4-2-1."),
            record(case_id=2, prediction="Sceglierei 4-5-1."),
        ],
        fallback_client=client,
    )

    json_path, markdown_path, predictions_path = write_fallback_eval_outputs(
        report,
        tmp_path,
    )

    assert json_path.name == "fallback_eval.json"
    assert markdown_path.name == "fallback_eval.md"
    assert predictions_path.name == "fallback_predictions.jsonl"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["cases"] == 2
    assert payload["fallback_used_count"] == 1
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["case_id"] == 1
    assert rows[0]["prediction"] == "Sceglierei 3-4-2-1."
    assert rows[0]["final_source"] == "primary"
    assert rows[1]["case_id"] == 2
    assert rows[1]["prediction"] == "Sceglierei 3-4-2-1 perche resta nel prompt."
    assert rows[1]["final_source"] == "fallback"
```

- [ ] **Step 2: Run writer tests and confirm RED**

Run:

```bash
python -m pytest tests/test_fallback_eval.py -q
```

Expected:

```text
ImportError: cannot import name 'render_fallback_eval_markdown'
```

- [ ] **Step 3: Add render/write functions**

Append to `src/fantabrain_llm/fallback_eval.py`:

```python
def render_fallback_eval_markdown(report: FallbackEvalReport) -> str:
    lines = [
        "# OpenAI Fallback Eval Report",
        "",
        f"Cases: {report.cases}",
        "",
        "## Summary",
        "",
        f"- fallback_used_count: {report.fallback_used_count}",
        f"- fallback_success_count: {report.fallback_success_count}",
        f"- unresolved_safe_count: {report.unresolved_safe_count}",
        f"- estimated_total_cost_usd: {report.estimated_total_cost_usd}",
        "",
        "## Primary Action Counts",
        "",
    ]
    lines.extend(_counter_lines(report.primary_action_counts))
    lines.extend(["", "## Final Source Counts", ""])
    lines.extend(_counter_lines(report.final_source_counts))
    lines.extend(["", "## Primary Violation Counts", ""])
    lines.extend(_counter_lines(report.primary_violation_counts))
    lines.extend(["", "## Final Violation Counts", ""])
    lines.extend(_counter_lines(report.final_violation_counts))
    lines.extend(["", "## Fallback Cases", ""])

    fallback_results = [result for result in report.results if result.fallback_used]
    if not fallback_results:
        lines.append("- No fallback cases")
    for result in fallback_results:
        lines.extend(
            [
                f"### Case {result.case_id}: {result.mode} / {result.task}",
                "",
                f"- Primary action: {result.primary_action}",
                f"- Fallback action: {result.fallback_action}",
                f"- Final source: {result.final_source.value}",
                f"- Estimated cost USD: {result.estimated_cost_usd}",
            ]
        )
        if result.primary_violations:
            lines.append("- Primary violations:")
            for violation in result.primary_violations:
                lines.append(f"  - {violation['check']}: `{violation['term']}`")
        else:
            lines.append("- Primary violations: none")
        if result.fallback_violations:
            lines.append("- Fallback violations:")
            for violation in result.fallback_violations:
                lines.append(f"  - {violation['check']}: `{violation['term']}`")
        else:
            lines.append("- Fallback violations: none")
        lines.append("")

    return "\n".join(lines)


def write_fallback_eval_outputs(
    report: FallbackEvalReport,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "fallback_eval.json"
    markdown_path = target / "fallback_eval.md"
    predictions_path = target / "fallback_predictions.jsonl"

    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(render_fallback_eval_markdown(report), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in report.results:
            handle.write(
                json.dumps(_fallback_prediction_row(result), ensure_ascii=False) + "\n"
            )

    return json_path, markdown_path, predictions_path


def _fallback_prediction_row(result: FallbackCaseResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "mode": result.mode,
        "task": result.task,
        "prompt": result.prompt,
        "expected": result.expected,
        "prediction": result.final_prediction,
        "primary_prediction": result.primary_prediction,
        "primary_action": result.primary_action,
        "fallback_used": result.fallback_used,
        "fallback_prediction": result.fallback_prediction,
        "fallback_action": result.fallback_action,
        "final_source": result.final_source.value,
        "estimated_cost_usd": result.estimated_cost_usd,
    }


def _counter_lines(counter: dict[str, int]) -> list[str]:
    if not counter:
        return ["- none: 0"]
    return [f"- {key}: {value}" for key, value in counter.items()]
```

- [ ] **Step 4: Run writer tests and confirm GREEN**

Run:

```bash
python -m pytest tests/test_fallback_eval.py -q
```

Expected:

```text
8 passed
```

- [ ] **Step 5: Commit Task 3**

```bash
git add src/fantabrain_llm/fallback_eval.py tests/test_fallback_eval.py
git commit -m "feat: write fallback eval reports"
```

---

### Task 4: Fallback Eval CLI

**Files:**
- Create: `scripts/run_fallback_eval.py`
- Test: `tests/test_run_fallback_eval_cli.py`

- [ ] **Step 1: Write failing CLI tests with injected fake client**

Create `tests/test_run_fallback_eval_cli.py`:

```python
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
```

- [ ] **Step 2: Run CLI tests and confirm RED**

Run:

```bash
python -m pytest tests/test_run_fallback_eval_cli.py -q
```

Expected:

```text
FileNotFoundError
```

or:

```text
No such file or directory: 'scripts/run_fallback_eval.py'
```

- [ ] **Step 3: Implement CLI**

Create `scripts/run_fallback_eval.py`:

```python
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
```

- [ ] **Step 4: Run CLI tests and confirm GREEN**

Run:

```bash
python -m pytest tests/test_run_fallback_eval_cli.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Run combined fallback tests**

Run:

```bash
python -m pytest tests/test_openai_fallback.py tests/test_fallback_eval.py tests/test_run_fallback_eval_cli.py -q
```

Expected:

```text
16 passed
```

- [ ] **Step 6: Commit Task 4**

```bash
git add scripts/run_fallback_eval.py tests/test_run_fallback_eval_cli.py
git commit -m "feat: add fallback eval cli"
```

---

### Task 5: Docs and Runbook

**Files:**
- Create: `docs/runbooks/openai-fallback-eval-v0.md`
- Modify: `README.md`

- [ ] **Step 1: Add runbook**

Create `docs/runbooks/openai-fallback-eval-v0.md`:

```markdown
# OpenAI Fallback Eval v0

This runbook evaluates the app-style fallback path offline:

```text
Qwen/LoRA prediction -> output filter -> optional gpt-5.4-mini fallback -> output filter -> final answer
```

## Inputs

- Existing prediction run with `predictions.jsonl`.
- `OPENAI_API_KEY` set in the shell or Colab secrets.

Do not commit API keys or `.env` files.

## Command

```bash
python scripts/run_fallback_eval.py \
  --predictions reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/predictions.jsonl \
  --output-dir reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0-fallback-eval-v0 \
  --fallback-model gpt-5.4-mini \
  --max-output-tokens 350 \
  --temperature 0.2
```

## Outputs

- `fallback_eval.json`
- `fallback_eval.md`
- `fallback_predictions.jsonl`

## Read The Result

The most important fields are:

- `fallback_used_count`: how often Qwen/LoRA failed hard enough to need fallback.
- `fallback_success_count`: how often fallback cleared the filter.
- `unresolved_safe_count`: how often both primary and fallback failed.
- `estimated_total_cost_usd`: rough fallback cost from API usage tokens.

## Product Gate

This is promising only if fallback is used on a minority of cases, most fallback answers pass the second filter, unresolved safe cases are rare, and manual review confirms the fallback answers are not generic or misleading.
```

- [ ] **Step 2: Add README command**

Add this section to `README.md` after the output filter command section:

```markdown
## OpenAI Fallback Eval

After generating a prediction run and filtering it, you can evaluate the app-style fallback path against the blocked cases only.

```bash
python scripts/run_fallback_eval.py \
  --predictions reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0/predictions.jsonl \
  --output-dir reports/runs/qwen25-3b-fantabrain-sft-v2-pagella-v0-fallback-eval-v0 \
  --fallback-model gpt-5.4-mini \
  --max-output-tokens 350 \
  --temperature 0.2
```

The command writes `fallback_eval.json`, `fallback_eval.md`, and `fallback_predictions.jsonl`. Tests mock the OpenAI call and never use the real API.
```

- [ ] **Step 3: Inspect docs**

Run:

```bash
rg "OPENAI_API_KEY|run_fallback_eval|fallback_eval" README.md docs/runbooks/openai-fallback-eval-v0.md
```

Expected:

```text
README.md
docs/runbooks/openai-fallback-eval-v0.md
```

- [ ] **Step 4: Commit Task 5**

```bash
git add README.md docs/runbooks/openai-fallback-eval-v0.md
git commit -m "docs: add openai fallback eval runbook"
```

---

### Task 6: Full Verification and Branch Handoff

**Files:**
- Verify all touched files.
- No new source files after this task.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
python -m pytest tests/test_openai_fallback.py tests/test_fallback_eval.py tests/test_run_fallback_eval_cli.py -q
```

Expected:

```text
16 passed
```

- [ ] **Step 2: Run output-filter regression tests**

Run:

```bash
python -m pytest tests/test_output_filter.py tests/test_filter_predictions_cli.py tests/test_prediction_audit.py -q
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 3: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 4: Run ruff on touched Python files**

Run:

```bash
python -m ruff check src/fantabrain_llm/openai_fallback.py src/fantabrain_llm/fallback_eval.py scripts/run_fallback_eval.py tests/test_openai_fallback.py tests/test_fallback_eval.py tests/test_run_fallback_eval_cli.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 5: Check for leaked secrets**

Run:

```bash
rg -n "sk-[A-Za-z0-9]{20,}|OPENAI_API_KEY=[A-Za-z0-9_]" src scripts tests docs README.md
```

Expected:

```text
No matches for real secrets. README/runbook may contain the literal variable name OPENAI_API_KEY, but not a real value.
```

- [ ] **Step 6: Review diff**

Run:

```bash
git diff --check origin/master...HEAD
git diff --stat origin/master...HEAD
```

Expected:

```text
No whitespace errors.
Diff includes only fallback eval client, eval core, CLI, tests, README, and runbook.
```

- [ ] **Step 7: Commit final cleanup if any verification fixes were needed**

If verification required edits, commit them:

```bash
git add src scripts tests docs README.md
git commit -m "test: verify openai fallback eval"
```

If no edits were required after Task 5, do not create an empty commit.

- [ ] **Step 8: Push branch**

Run:

```bash
git push -u origin codex/openai-fallback-eval-v0
```

Expected:

```text
Branch codex/openai-fallback-eval-v0 is pushed.
```

- [ ] **Step 9: Update `llm-memory` after implementation is complete**

Append a short entry to:

```text
C:\Users\DantePagani\llm-memory\wiki\projects\fantabrain-llm\project-overview.md
```

Entry content:

```markdown
## OpenAI Fallback Eval v0 - 2026-06-23

Codex implemented the offline OpenAI fallback evaluation branch `codex/openai-fallback-eval-v0`. The feature filters Qwen/LoRA predictions, calls `gpt-5.4-mini` only for blocked primary outputs, filters fallback answers, writes `fallback_eval.json`, `fallback_eval.md`, and `fallback_predictions.jsonl`, and estimates fallback cost from API usage tokens. Tests mock OpenAI and do not make real API calls.
```

---

## Self-Review

Spec coverage:

- Offline fallback over existing `predictions.jsonl`: Task 2 and Task 4.
- Official OpenAI Responses API client: Task 1.
- Fallback only on `fallback` and `safe`: Task 2 tests and implementation.
- Second filter pass on fallback output: Task 2.
- Conservative safe response if fallback fails: Task 2.
- Outputs `fallback_eval.json`, `fallback_eval.md`, `fallback_predictions.jsonl`: Task 3 and Task 4.
- Usage tokens and cost estimate: Task 1 and Task 2.
- Missing key/provider errors are clear: Task 1 and Task 4.
- No real API calls in tests: Task 1 tests monkeypatch HTTP, Task 4 injects fake client.
- No app integration, no training, no dataset changes: file structure excludes app/dataset/training files.

Placeholder scan:

- The plan does not contain unresolved implementation placeholders.

Type consistency:

- `FallbackResponse`, `FallbackUsage`, `OpenAIFallbackClient`, `FinalSource`, `FallbackCaseResult`, `FallbackEvalReport`, `run_fallback_eval`, `render_fallback_eval_markdown`, and `write_fallback_eval_outputs` are defined before later tasks reference them.
- CLI `client_factory` signature matches `OpenAIFallbackClient(model=..., max_output_tokens=..., temperature=...)`.
- `FinalSource` enum values match report values: `primary`, `fallback`, `safe`.
