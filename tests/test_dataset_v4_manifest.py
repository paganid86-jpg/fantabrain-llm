from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_v4_manifest_uses_v2_as_base_not_v3() -> None:
    manifest_path = ROOT / "datasets" / "v4" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    assert manifest["version"] == "v4"
    assert manifest["base_dataset"] == "datasets/v2/train.jsonl"
    assert manifest["train_path"] == "datasets/v4/train.jsonl"
    assert manifest["p3bis_examples"] == 20
    assert manifest["p3bis_balance"]["by_mode"] == {"mantra": 10, "classic": 10}
    assert manifest["quality_gates"]["source"] == "v4_manual"
    assert manifest["quality_gates"]["forbid_dataset_v3_as_base"] is True


def test_dataset_v4_manifest_keeps_p3bis_blocks_balanced() -> None:
    manifest_path = ROOT / "datasets" / "v4" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    block_total = sum(block["examples"] for block in manifest["p3bis_blocks"])
    assert block_total == 20

    mode_totals = {
        "mantra": sum(block["mode_split"]["mantra"] for block in manifest["p3bis_blocks"]),
        "classic": sum(block["mode_split"]["classic"] for block in manifest["p3bis_blocks"]),
    }
    assert mode_totals == {"mantra": 10, "classic": 10}

    block_names = {block["name"] for block in manifest["p3bis_blocks"]}
    assert block_names == {
        "mantra_anti_leak",
        "classic_anti_leak",
        "decision_inversion",
        "refusal_stop",
    }


def test_dataset_v4_manifest_tracks_repair_and_promotion_targets() -> None:
    manifest_path = ROOT / "datasets" / "v4" / "manifest.yaml"

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    repair_targets = manifest["repair_targets"]
    assert set(repair_targets["mantra_cases"]) == {2, 3, 4, 10, 11, 20, 25, 27, 38}
    assert set(repair_targets["classic_cases"]) == {13, 15, 16, 21, 24, 28, 29, 30, 36, 40}
    assert set(repair_targets["preserve_signals"]) == {6, 9, 32, 34, 37}

    promotion = manifest["promotion_gates"]
    assert promotion["effective_average_gt"] == 2.69
    assert promotion["hallucination_free_gt"] == 26
    assert promotion["raw_average_min"] == 3.10
    assert promotion["case_2_no_invented_modules"] is True
    assert promotion["mantra_forbid_modificatore"] is True
    assert promotion["classic_forbid_mantra_vocabulary"] is True
