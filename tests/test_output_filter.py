from __future__ import annotations

from fantabrain_llm.output_filter import FilterAction, filter_model_output


def filter_output(
    *,
    mode: str,
    prompt: str,
    prediction: str,
    fallback_failed: bool = False,
    task: str = "lineup_advice",
):
    return filter_model_output(
        mode=mode,
        task=task,
        prompt=prompt,
        prediction=prediction,
        fallback_failed=fallback_failed,
    )


def test_clean_mantra_output_passes_without_violations() -> None:
    decision = filter_output(
        mode="mantra",
        prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
        prediction="Sceglierei 3-4-2-1 se hai copertura sugli esterni e due T sicure.",
    )

    assert decision.action is FilterAction.PASS
    assert decision.reason == "no_violations"
    assert decision.violations == []


def test_malformed_only_classic_output_passes_with_warnings() -> None:
    decision = filter_output(
        mode="classic",
        prompt="Modalita Classic. Sono primo.",
        prediction="Sceglierei il titolare, ma evita una scelta offENSIVO.",
    )

    assert decision.action is FilterAction.PASS_WITH_WARNINGS
    assert decision.reason == "soft_violations"
    assert [violation.check for violation in decision.violations] == ["malformed_terms"]


def test_mantra_forbidden_term_triggers_fallback() -> None:
    decision = filter_output(
        mode="mantra",
        prompt="Modalita Mantra. Meglio rischiare un centrocampista o coprirmi?",
        prediction="Eviterei il modificatore e sceglierei il profilo piu stabile.",
    )

    assert decision.action is FilterAction.FALLBACK
    assert decision.reason == "hard_violations"
    assert [violation.check for violation in decision.violations] == ["mantra_forbidden_terms"]


def test_mantra_invented_module_triggers_fallback() -> None:
    decision = filter_output(
        mode="mantra",
        prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
        prediction="Sceglierei 4-5-1 per proteggere meglio il centrocampo.",
    )

    assert decision.action is FilterAction.FALLBACK
    assert decision.reason == "hard_violations"
    assert [violation.check for violation in decision.violations] == ["invented_modules"]


def test_classic_standalone_role_code_leakage_triggers_fallback() -> None:
    decision = filter_output(
        mode="classic",
        prompt="Modalita Classic. Chi metto capitano?",
        prediction="La T e una scelta forte, ma sceglierei il rigorista.",
    )

    assert decision.action is FilterAction.FALLBACK
    assert decision.reason == "hard_violations"
    assert [violation.check for violation in decision.violations] == [
        "classic_role_code_leakage"
    ]


def test_empty_prediction_returns_safe_without_violations() -> None:
    decision = filter_output(
        mode="mantra",
        prompt="Modalita Mantra. Chi schiero?",
        prediction="  \n\t  ",
    )

    assert decision.action is FilterAction.SAFE
    assert decision.reason == "empty_prediction"
    assert decision.violations == []


def test_failed_fallback_after_hard_violation_returns_safe() -> None:
    decision = filter_output(
        mode="mantra",
        prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
        prediction="Sceglierei 4-5-1 per proteggere meglio il centrocampo.",
        fallback_failed=True,
    )

    assert decision.action is FilterAction.SAFE
    assert decision.reason == "fallback_failed_hard_violations"
    assert [violation.check for violation in decision.violations] == ["invented_modules"]
