from __future__ import annotations

import json
from pathlib import Path

from fantabrain_llm.output_filter import (
    FilterAction,
    filter_model_output,
    filter_prediction_records,
    render_filter_markdown,
    write_filter_outputs,
)


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


def prediction_record(
    *,
    case_id: int,
    mode: str,
    task: str,
    prompt: str,
    prediction: str,
    expected: str = "Expected answer.",
    provider: str = "echo",
    model: str = "echo-baseline",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "mode": mode,
        "task": task,
        "prompt": prompt,
        "prediction": prediction,
        "expected": expected,
        "provider": provider,
        "model": model,
    }


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


def test_filter_prediction_records_counts_actions() -> None:
    report = filter_prediction_records(
        [
            prediction_record(
                case_id=1,
                mode="mantra",
                task="lineup_advice",
                prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
                prediction="Sceglierei 3-4-2-1 se hai copertura sugli esterni.",
            ),
            prediction_record(
                case_id=2,
                mode="mantra",
                task="lineup_advice",
                prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
                prediction="Sceglierei 4-5-1 per proteggere meglio il centrocampo.",
            ),
        ]
    )

    assert report.cases == 2
    assert report.decision_counts == {"pass": 1, "fallback": 1}
    assert report.violation_counts == {"invented_modules": 1}
    assert report.results[1].decision.action is FilterAction.FALLBACK


def test_filter_prediction_records_preserves_empty_prediction_safe_decision() -> None:
    report = filter_prediction_records(
        [
            prediction_record(
                case_id=1,
                mode="mantra",
                task="lineup_advice",
                prompt="Modalita Mantra. Chi schiero?",
                prediction="  \n\t  ",
            )
        ]
    )

    assert report.cases == 1
    assert report.decision_counts == {"safe": 1}
    assert report.violation_counts == {}
    assert report.results[0].decision.action is FilterAction.SAFE
    assert report.results[0].decision.reason == "empty_prediction"


def test_render_filter_markdown_includes_summary_and_cases() -> None:
    report = filter_prediction_records(
        [
            prediction_record(
                case_id=2,
                mode="mantra",
                task="lineup_advice",
                prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
                prediction="Sceglierei 4-5-1 per proteggere meglio il centrocampo.",
            )
        ]
    )

    markdown = render_filter_markdown(report)

    assert "# Output Filter Report" in markdown
    assert "fallback: 1" in markdown
    assert "Case 2" in markdown
    assert "invented_modules" in markdown


def test_write_filter_outputs_writes_json_and_markdown(tmp_path: Path) -> None:
    report = filter_prediction_records(
        [
            prediction_record(
                case_id=1,
                mode="mantra",
                task="lineup_advice",
                prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
                prediction="Sceglierei 3-4-2-1 se hai copertura sugli esterni.",
            )
        ]
    )

    json_path, markdown_path = write_filter_outputs(report, tmp_path)

    assert json_path.name == "output_filter.json"
    assert markdown_path.name == "output_filter.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["cases"] == 1
    assert payload["decision_counts"] == {"pass": 1}
