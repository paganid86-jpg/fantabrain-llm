from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_v2_manifest_keeps_p2_balanced() -> None:
    manifest_path = ROOT / "datasets" / "v2" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    assert manifest["version"] == "v2"
    assert manifest["base_dataset"] == "datasets/v1/train.jsonl"
    assert manifest["train_path"] == "datasets/v2/train.jsonl"
    assert manifest["p2_examples"] == 80
    assert manifest["p2_balance"]["by_mode"] == {"mantra": 40, "classic": 40}

    block_total = sum(block["examples"] for block in manifest["p2_blocks"])
    assert block_total == 80

    mode_totals = {
        "mantra": sum(block["mode_split"]["mantra"] for block in manifest["p2_blocks"]),
        "classic": sum(block["mode_split"]["classic"] for block in manifest["p2_blocks"]),
    }
    assert mode_totals == manifest["p2_balance"]["by_mode"]


def test_dataset_v2_manifest_tracks_repair_targets() -> None:
    manifest_path = ROOT / "datasets" / "v2" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    target_cases = set(manifest["repair_targets"]["primary_cases"])
    assert target_cases == {2, 3, 6, 9, 20, 27, 28, 34, 39}
    assert set(manifest["repair_targets"]["secondary_cases"]) == {7, 15, 31, 35, 36}

    block_names = {block["name"] for block in manifest["p2_blocks"]}
    assert block_names == {
        "classic_modificatore",
        "mantra_role_codes_guardrail",
        "risk_varianza_decisioni",
        "refusal_grounded_clean",
        "italiano_asciutto_decision_first",
    }

    quality_gates = manifest["quality_gates"]
    assert quality_gates["min_quality_score"] == 5
    assert quality_gates["source"] == "v2_manual"
    assert quality_gates["no_pagella_training"] is True
    assert quality_gates["forbid_eval_prompt_leakage"] is True
    assert quality_gates["forbid_real_player_names"] is True
    assert quality_gates["forbid_specific_live_facts"] is True
    assert quality_gates["forbid_invented_percentages"] is True
    assert quality_gates["forbid_specific_scores_or_votes"] is True
    assert quality_gates["forbid_fake_rules"] is True
    assert quality_gates["forbid_role_code_corruption"] is True
    assert quality_gates["keep_mantra_classic_same_level"] is True
