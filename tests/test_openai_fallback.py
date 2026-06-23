from __future__ import annotations

import json
import urllib.error

import pytest

from fantabrain_llm.openai_fallback import (
    DEFAULT_FALLBACK_MODEL,
    OPENAI_RESPONSES_URL,
    OpenAIFallbackClient,
    OpenAIFallbackError,
    estimate_cost_usd,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def close(self) -> None:
        return None


class FakeRawHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> FakeRawHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_missing_api_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAIFallbackError, match="OPENAI_API_KEY"):
        OpenAIFallbackClient()


def test_estimate_cost_for_default_model() -> None:
    assert estimate_cost_usd(1000, 500) == pytest.approx(0.003)
    assert estimate_cost_usd(None, 500) is None
    assert estimate_cost_usd(1000, None) is None
    assert estimate_cost_usd(1000, 500, model="other-model") is None


def test_generate_builds_payload_and_parses_output_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse(
            {
                "model": DEFAULT_FALLBACK_MODEL,
                "output_text": "Sceglierei il centrocampista piu solido.",
                "usage": {"input_tokens": 1000, "output_tokens": 500},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAIFallbackClient(api_key="sk-test", timeout_seconds=12)
    response = client.generate(mode="mantra", task="lineup_advice", prompt="Chi schiero?")

    assert captured["url"] == OPENAI_RESPONSES_URL
    assert captured["method"] == "POST"
    assert captured["timeout"] == 12
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["headers"]["Content-type"] == "application/json"
    payload = captured["payload"]
    assert payload["model"] == DEFAULT_FALLBACK_MODEL
    assert isinstance(payload["instructions"], str)
    assert payload["max_output_tokens"] == 350
    assert payload["temperature"] == 0.2
    assert payload["tools"] == []
    assert payload["store"] is False
    assert payload["input"] == [
        {
            "role": "user",
            "content": "Modalita: mantra\nTask: lineup_advice\nPrompt utente:\nChi schiero?",
        }
    ]
    assert response.text == "Sceglierei il centrocampista piu solido."
    assert response.model == DEFAULT_FALLBACK_MODEL
    assert response.usage.input_tokens == 1000
    assert response.usage.output_tokens == 500
    assert response.usage.estimated_cost_usd == pytest.approx(0.003)
    assert response.to_dict()["usage"]["estimated_cost_usd"] == pytest.approx(0.003)


def test_generate_parses_nested_output_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
        del request, timeout
        return FakeHTTPResponse(
            {
                "model": DEFAULT_FALLBACK_MODEL,
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": " Eviterei il rischio inutile. "}
                        ]
                    }
                ],
                "usage": {"input_tokens": 200, "output_tokens": 100},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAIFallbackClient(api_key="sk-test")
    response = client.generate(mode="classic", task="trade_advice", prompt="Scambio?")

    assert response.text == "Eviterei il rischio inutile."
    assert response.usage.estimated_cost_usd == pytest.approx(0.0006)


def test_http_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
        del request, timeout
        raise urllib.error.HTTPError(
            url=OPENAI_RESPONSES_URL,
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=FakeHTTPResponse({"error": {"message": "rate limited"}}),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAIFallbackClient(api_key="sk-test")
    with pytest.raises(OpenAIFallbackError, match="rate limited"):
        client.generate(mode="mantra", task="lineup_advice", prompt="Chi schiero?")


def test_timeout_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
        del request, timeout
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAIFallbackClient(api_key="sk-test")
    with pytest.raises(OpenAIFallbackError, match="timed out"):
        client.generate(mode="mantra", task="lineup_advice", prompt="Chi schiero?")


def test_invalid_response_bytes_are_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: int) -> FakeRawHTTPResponse:
        del request, timeout
        return FakeRawHTTPResponse(b"\xff")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAIFallbackClient(api_key="sk-test")
    with pytest.raises(OpenAIFallbackError, match="decode"):
        client.generate(mode="mantra", task="lineup_advice", prompt="Chi schiero?")


def test_malformed_response_without_text_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
        del request, timeout
        return FakeHTTPResponse({"model": DEFAULT_FALLBACK_MODEL, "output": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAIFallbackClient(api_key="sk-test")
    with pytest.raises(OpenAIFallbackError, match="No text output"):
        client.generate(mode="mantra", task="lineup_advice", prompt="Chi schiero?")
