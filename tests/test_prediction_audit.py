from __future__ import annotations

from fantabrain_llm.prediction_audit import (
    audit_prediction_records,
    extract_modules,
    render_audit_markdown,
)


def prediction(
    *,
    case_id: int,
    mode: str,
    prompt: str,
    text: str,
    task: str = "lineup_advice",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "mode": mode,
        "task": task,
        "tags": ["test"],
        "prompt": prompt,
        "expected": "Gold",
        "prediction": text,
        "provider": "echo",
        "model": "echo",
    }


def test_extract_modules_uses_full_module_tokens() -> None:
    assert extract_modules("Meglio 3-4-2-1 o 4-3-3?") == {"3-4-2-1", "4-3-3"}
    assert "3-4-2" not in extract_modules("Meglio 3-4-2-1?")


def test_audit_flags_mantra_forbidden_terms_and_invented_modules() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=2,
                mode="mantra",
                prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
                text="Sceglierei 4-5-1. Il modificatore aiuta.",
            )
        ]
    )

    assert report.summary == {"mantra_forbidden_terms": 1, "invented_modules": 1}
    assert report.hard_violation_count == 2
    assert {violation.check for violation in report.violations} == {
        "invented_modules",
        "mantra_forbidden_terms",
    }


def test_audit_flags_mantra_module_invention_when_prompt_has_no_modules() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=4,
                mode="mantra",
                prompt="Modalita Mantra. Ho pochi esterni: che approccio uso?",
                text="Passerei al 4-3-3 per aumentare copertura.",
            )
        ]
    )

    assert report.summary == {"invented_modules": 1}
    assert report.violations[0].term == "4-3-3"
    assert report.hard_violation_count == 1


def test_audit_allows_classic_role_codes_present_in_prompt() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=1,
                mode="classic",
                prompt="Classic ma sto confrontando la vecchia T Mantra con un attaccante.",
                text="La T citata nel prompt non e leakage nuovo.",
            )
        ]
    )

    assert report.violations == []


def test_audit_flags_classic_role_code_leakage_token_aware() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=7,
                mode="classic",
                prompt="Modalita Classic. Chi metto capitano?",
                text="La T e una scelta forte, ma titolare e centrale non sono codici.",
            )
        ]
    )

    assert [violation.check for violation in report.violations] == ["classic_role_code_leakage"]
    assert report.violations[0].term == "T"
    assert report.hard_violation_count == 1


def test_audit_avoids_common_italian_single_letter_false_positive() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=11,
                mode="classic",
                prompt="Modalita Classic. Chi metto a centrocampo?",
                text="A parita di titolarita sceglierei il profilo con piu bonus.",
            )
        ]
    )

    assert report.violations == []


def test_audit_flags_classic_module_language_as_soft_violation() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=8,
                mode="classic",
                prompt="Modalita Classic. Chi metto in difesa?",
                text="Valuta lo slot e il modulo principale prima della scelta.",
            )
        ]
    )

    assert report.summary == {"classic_module_language": 2}
    assert report.hard_violation_count == 0


def test_audit_flags_malformed_terms_as_soft_violations() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=35,
                mode="classic",
                prompt="Modalita Classic. Sono primo.",
                text="La formazione offENSIVO conserva evita due punteggiere.",
            )
        ]
    )

    assert report.summary == {"malformed_terms": 2}
    assert report.hard_violation_count == 0


def test_audit_clean_record_has_no_violations() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=10,
                mode="mantra",
                prompt="Modalita Mantra. Meglio 3-4-2-1 o 4-3-3?",
                text="Sceglierei 3-4-2-1 se hai copertura sugli esterni e due T sicure.",
            )
        ]
    )

    assert report.summary == {}
    assert report.violations == []
    assert report.hard_violation_count == 0


def test_render_audit_markdown_includes_summary_and_case_details() -> None:
    report = audit_prediction_records(
        [
            prediction(
                case_id=3,
                mode="mantra",
                prompt="Modalita Mantra. Che faccio?",
                text="Usa il modificatore.",
            )
        ]
    )

    markdown = render_audit_markdown(report)

    assert "# Prediction Audit" in markdown
    assert "mantra_forbidden_terms: 1" in markdown
    assert "Case 3" in markdown
    assert "modificatore" in markdown
    assert report.to_dict()["hard_violation_count"] == 1
