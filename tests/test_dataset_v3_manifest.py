from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_v3_manifest_keeps_p3_balanced() -> None:
    manifest_path = ROOT / "datasets" / "v3" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    assert manifest["version"] == "v3"
    assert manifest["base_dataset"] == "datasets/v2/train.jsonl"
    assert manifest["train_path"] == "datasets/v3/train.jsonl"
    assert manifest["p3_examples"] == 40
    assert manifest["p3_balance"]["by_mode"] == {"mantra": 20, "classic": 20}

    block_total = sum(block["examples"] for block in manifest["p3_blocks"])
    assert block_total == 40

    mode_totals = {
        "mantra": sum(block["mode_split"]["mantra"] for block in manifest["p3_blocks"]),
        "classic": sum(block["mode_split"]["classic"] for block in manifest["p3_blocks"]),
    }
    assert mode_totals == manifest["p3_balance"]["by_mode"]


def test_dataset_v3_manifest_tracks_cleanup_targets() -> None:
    manifest_path = ROOT / "datasets" / "v3" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    target_cases = set(manifest["repair_targets"]["primary_cases"])
    assert target_cases == {2, 5, 6, 7, 9, 10, 19, 25, 27, 28, 29, 32, 34, 37}
    assert set(manifest["repair_targets"]["secondary_cases"]) == {15, 31, 36, 40}

    block_names = {block["name"] for block in manifest["p3_blocks"]}
    assert block_names == {
        "mantra_no_module_invention",
        "classic_modificatore_clean",
        "refusal_stop_clean",
        "mantra_roles_no_cross_mode",
        "italiano_cleanup_decision_first",
    }

    quality_gates = manifest["quality_gates"]
    assert quality_gates["min_quality_score"] == 5
    assert quality_gates["source"] == "v3_manual"
    assert quality_gates["no_pagella_training"] is True
    assert quality_gates["forbid_eval_prompt_leakage"] is True
    assert quality_gates["forbid_real_player_names"] is True
    assert quality_gates["forbid_specific_live_facts"] is True
    assert quality_gates["forbid_invented_modules"] is True
    assert quality_gates["forbid_cross_mode_vocabulary"] is True
    assert quality_gates["forbid_broken_generated_words"] is True
    assert quality_gates["keep_mantra_classic_same_level"] is True
