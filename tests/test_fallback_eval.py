from __future__ import annotations

import pytest

from fantabrain_llm.fallback_eval import (
    SAFE_FALLBACK_RESPONSE,
    FallbackEvalError,
    FinalSource,
    run_fallback_eval,
)
from fantabrain_llm.openai_fallback import FallbackResponse, FallbackUsage


def prediction_record(
    *,
    case_id: int = 1,
    mode: str = "mantra",
    task: str = "lineup_advice",
    prompt: str = "Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
    prediction: str = "Sceglierei 3-4-2-1 se hai copertura sugli esterni.",
    expected: str | None = "Expected answer.",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "mode": mode,
        "task": task,
        "prompt": prompt,
        "prediction": prediction,
        "expected": expected,
    }


def fallback_response(
    text: str,
    *,
    input_tokens: int | None = 10,
    output_tokens: int | None = 20,
    estimated_cost_usd: float | None = 0.001,
) -> FallbackResponse:
    return FallbackResponse(
        text=text,
        model="fake-fallback",
        usage=FallbackUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
        ),
    )


class FakeFallbackClient:
    def __init__(self, *responses: FallbackResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, str]] = []

    def generate(self, *, mode: str, task: str, prompt: str) -> FallbackResponse:
        self.calls.append({"mode": mode, "task": task, "prompt": prompt})
        return self.responses.pop(0)


def test_primary_pass_does_not_call_fallback() -> None:
    client = FakeFallbackClient(fallback_response("unused"))

    report = run_fallback_eval([prediction_record()], fallback_client=client)

    assert client.calls == []
    result = report.results[0]
    assert result.final_source is FinalSource.PRIMARY
    assert result.final_prediction == "Sceglierei 3-4-2-1 se hai copertura sugli esterni."
    assert result.fallback_used is False
    assert result.fallback_prediction is None


def test_primary_pass_with_warnings_does_not_call_fallback() -> None:
    client = FakeFallbackClient(fallback_response("unused"))

    report = run_fallback_eval(
        [
            prediction_record(
                mode="classic",
                prompt="Modalita Classic. Sono primo.",
                prediction="Sceglierei il titolare, ma evita una scelta offENSIVO.",
            )
        ],
        fallback_client=client,
    )

    assert client.calls == []
    result = report.results[0]
    assert result.primary_action == "pass_with_warnings"
    assert result.final_source is FinalSource.PRIMARY


def test_primary_fallback_calls_fallback_and_clean_fallback_becomes_final() -> None:
    client = FakeFallbackClient(
        fallback_response("Sceglierei 3-4-2-1, restando sui moduli citati.")
    )

    report = run_fallback_eval(
        [
            prediction_record(
                prediction="Sceglierei 4-5-1 per proteggere meglio il centrocampo.",
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
    assert result.final_prediction == "Sceglierei 3-4-2-1, restando sui moduli citati."
    assert result.usage == FallbackUsage(
        input_tokens=10,
        output_tokens=20,
        estimated_cost_usd=0.001,
    )


def test_primary_safe_empty_output_calls_fallback() -> None:
    client = FakeFallbackClient(fallback_response("Mi mancano dati: scegli il titolare certo."))

    report = run_fallback_eval(
        [prediction_record(prediction="  \n\t  ")],
        fallback_client=client,
    )

    assert len(client.calls) == 1
    result = report.results[0]
    assert result.primary_action == "safe"
    assert result.final_source is FinalSource.FALLBACK
    assert result.final_prediction == "Mi mancano dati: scegli il titolare certo."


def test_fallback_hard_failure_becomes_safe_response() -> None:
    client = FakeFallbackClient(
        fallback_response("Sceglierei 4-5-1 anche se il prompt non lo conferma.")
    )

    report = run_fallback_eval(
        [prediction_record(prediction="Eviterei il modificatore in questa scelta.")],
        fallback_client=client,
    )

    result = report.results[0]
    assert result.fallback_action == "safe"
    assert result.final_source is FinalSource.SAFE
    assert result.final_prediction == SAFE_FALLBACK_RESPONSE


def test_summary_counts_actions_sources_violations_and_cost() -> None:
    client = FakeFallbackClient(
        fallback_response(
            "Sceglierei 3-4-2-1, restando sui moduli citati.",
            estimated_cost_usd=0.001,
        ),
        fallback_response(
            "Sceglierei 4-5-1 anche se non e citato.",
            estimated_cost_usd=None,
        ),
    )

    report = run_fallback_eval(
        [
            prediction_record(case_id=10),
            prediction_record(
                case_id=42,
                prediction="Sceglierei 4-5-1 per proteggere meglio il centrocampo.",
            ),
            prediction_record(
                case_id=77,
                prediction="Eviterei il modificatore in questa scelta.",
            ),
        ],
        fallback_client=client,
    )

    assert report.cases == 3
    assert report.primary_action_counts == {"pass": 1, "fallback": 2}
    assert report.fallback_used_count == 2
    assert report.fallback_success_count == 1
    assert report.final_source_counts == {"primary": 1, "fallback": 1, "safe": 1}
    assert report.unresolved_safe_count == 1
    assert report.primary_violation_counts == {
        "invented_modules": 1,
        "mantra_forbidden_terms": 1,
    }
    assert report.final_violation_counts == {"invented_modules": 1}
    assert report.estimated_total_cost_usd == 0.001

    safe_result = report.results[2]
    assert safe_result.case_id == 77
    assert safe_result.fallback_violations[0]["case_id"] == 77
    assert report.to_dict()["final_source_counts"] == {
        "primary": 1,
        "fallback": 1,
        "safe": 1,
    }
    assert report.to_dict()["results"][1]["final_source"] == "fallback"
    assert report.to_dict()["results"][1]["usage"]["estimated_cost_usd"] == 0.001


def test_invalid_field_raises_fallback_eval_error() -> None:
    client = FakeFallbackClient(fallback_response("unused"))

    with pytest.raises(FallbackEvalError, match="case_id"):
        run_fallback_eval(
            [prediction_record(case_id=True)],  # type: ignore[arg-type]
            fallback_client=client,
        )
