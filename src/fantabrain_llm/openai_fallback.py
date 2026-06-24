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
Se manca contesto, dichiaralo e dai solo un criterio prudente.

Mantra: ragiona con ruoli, slot, coperture, moduli citati e panchina compatibile.
Non usare modificatore o reparto.
Classic: ragiona con reparti, titolarita, bonus/malus, modificatore difesa se pertinente.
Usa la panchina per reparto.
Non usare codici ruolo Mantra se non citati."""


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
    input_tokens: int | None,
    output_tokens: int | None,
    model: str = DEFAULT_FALLBACK_MODEL,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    if not _uses_gpt54_mini_pricing(model):
        return None
    return (
        (input_tokens / 1_000_000) * GPT54_MINI_INPUT_USD_PER_MTOK
        + (output_tokens / 1_000_000) * GPT54_MINI_OUTPUT_USD_PER_MTOK
    )


def _uses_gpt54_mini_pricing(model: str) -> bool:
    return model == DEFAULT_FALLBACK_MODEL or model.startswith(
        f"{DEFAULT_FALLBACK_MODEL}-"
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
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise OpenAIFallbackError("OPENAI_API_KEY is required for fallback evaluation.")

        self.model = model
        self.api_key = resolved_key
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate(self, *, mode: str, task: str, prompt: str) -> FallbackResponse:
        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=json.dumps(self._build_payload(mode=mode, task=task, prompt=prompt)).encode(
                "utf-8"
            ),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise OpenAIFallbackError(_format_http_error(exc)) from exc
        except urllib.error.URLError as exc:
            raise OpenAIFallbackError(f"OpenAI fallback request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OpenAIFallbackError(f"OpenAI fallback request timed out: {exc}") from exc
        except OSError as exc:
            raise OpenAIFallbackError(f"OpenAI fallback request failed: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise OpenAIFallbackError(
                f"OpenAI fallback response decode failed: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise OpenAIFallbackError("OpenAI fallback returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise OpenAIFallbackError("OpenAI fallback returned malformed JSON.")

        text = _extract_output_text(payload)
        model = str(payload.get("model") or self.model)
        return FallbackResponse(
            text=text,
            model=model,
            usage=_extract_usage(payload, model),
        )

    def _build_payload(self, *, mode: str, task: str, prompt: str) -> dict[str, object]:
        return {
            "model": self.model,
            "instructions": FALLBACK_SYSTEM_INSTRUCTIONS,
            "input": [
                {
                    "role": "user",
                    "content": (
                        f"Modalita: {mode}\n"
                        f"Task: {task}\n"
                        "Prompt utente:\n"
                        f"{prompt.strip()}"
                    ),
                }
            ],
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "tools": [],
            "store": False,
        }


def _extract_usage(payload: dict[str, Any], model: str) -> FallbackUsage:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return FallbackUsage(
            input_tokens=None,
            output_tokens=None,
            estimated_cost_usd=None,
        )

    input_tokens = _optional_int(usage.get("input_tokens"))
    output_tokens = _optional_int(usage.get("output_tokens"))
    return FallbackUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost_usd(input_tokens, output_tokens, model=model),
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
    message = None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = {}

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        message = error["message"]

    detail = message or body or exc.reason
    return f"OpenAI fallback request failed with HTTP {exc.code}: {detail}"
