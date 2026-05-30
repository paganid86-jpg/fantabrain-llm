from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_v1_manifest_keeps_mantra_classic_balanced() -> None:
    manifest_path = ROOT / "datasets" / "v1" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    assert manifest["target_examples"] == 320
    assert manifest["balance"]["by_mode"] == {"mantra": 160, "classic": 160}

    by_task_mode = manifest["balance"]["by_task_mode"]
    mantra_total = sum(task_counts["mantra"] for task_counts in by_task_mode.values())
    classic_total = sum(task_counts["classic"] for task_counts in by_task_mode.values())
    assert mantra_total == 160
    assert classic_total == 160


def test_dataset_v1_manifest_tracks_p1_repair_targets() -> None:
    manifest_path = ROOT / "datasets" / "v1" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    focus = manifest["p1_focus"]
    assert focus["mantra_role_codes"]["target_examples"] >= 120
    assert set(focus["mantra_role_codes"]["codes"]) == {"E", "M", "T", "W", "Pc", "Dc", "A"}
    assert focus["classic_specificity"]["target_examples"] >= 120
    assert focus["refusal_grounding"]["target_examples"] >= 50
    assert focus["short_binary_decisions"]["target_examples"] >= 50
    assert focus["core_vocab"]["target_examples"] >= 100

    quality_gates = manifest["quality_gates"]
    assert quality_gates["no_pagella_training"] is True
    assert quality_gates["forbid_real_player_names"] is True
    assert quality_gates["forbid_invented_percentages"] is True
    assert quality_gates["keep_mantra_classic_same_level"] is True
