from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.assemble_dataset_v5 import AssemblyError, assemble_dataset


SYSTEM = "Sei il coach AI privato di FantaBrain."


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def example(
    *,
    mode: str,
    task: str,
    source: str,
    user: str,
    assistant: str,
    tags: list[str],
    quality_score: int = 5,
) -> dict[str, object]:
    return {
        "mode": mode,
        "task": task,
        "source": source,
        "quality_score": quality_score,
        "tags": tags,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def write_manifest(
    path: Path,
    block_path: Path,
    *,
    examples: int = 2,
    mode_split: dict[str, int] | None = None,
    p4_balance: dict[str, int] | None = None,
    final_balance: dict[str, int] | None = None,
    blind_eval_path: str = "benchmarks/pagella_v0.jsonl",
) -> None:
    split = mode_split or {"mantra": 1, "classic": 1}
    payload = {
        "version": "v5",
        "base_dataset": "base.jsonl",
        "train_path": "train.jsonl",
        "blind_eval_path": blind_eval_path,
        "p4_examples": examples,
        "p4_balance": {"by_mode": p4_balance or {"mantra": 1, "classic": 1}},
        "final_balance": {"by_mode": final_balance or {"mantra": 2, "classic": 2}},
        "p4_blocks": [
            {
                "name": "test_block",
                "path": str(block_path),
                "examples": examples,
                "mode_split": split,
                "focus_tag": "test_focus",
                "target_cases": [2],
            }
        ],
        "quality_gates": {
            "min_quality_score": 5,
            "source": "v5_manual",
            "forbid_eval_prompt_leakage": True,
            "forbid_train_prompt_duplicates": True,
            "forbid_dataset_v3_as_base": True,
            "forbid_dataset_v4_as_base": True,
        },
        "mode_vocabulary_gates": {
            "mantra_forbidden_terms": ["modificatore", "reparto"],
            "classic_forbidden_terms": ["slot", "incastro", "codice", "mantra"],
            "classic_forbidden_role_codes": ["Pc", "T", "W", "M", "C", "E", "Dc", "Ds", "Dd"],
            "broken_terms": ["offENSIVO", "sicurata", "punteggianza"],
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def valid_mantra(user: str = "P4 Mantra prompt?") -> dict[str, object]:
    return example(
        mode="mantra",
        task="lineup_advice",
        source="v5_manual",
        user=user,
        assistant="Sceglierei il modulo gia coperto nei ruoli indicati. Se hai un dubbio tra talento e copertura, parto dalla stabilita dello slot raro e poi cerco bonus negli spazi flessibili. Mi manca la lista completa dei ruoli.",
        tags=["v5", "train", "mantra", "lineup_advice", "test_focus"],
    )


def valid_classic(user: str = "P4 Classic prompt?") -> dict[str, object]:
    return example(
        mode="classic",
        task="lineup_advice",
        source="v5_manual",
        user=user,
        assistant="Sceglierei il reparto con piu titolari sicuri. Se il bonus atteso e simile, proteggo voto medio e panchina prima della scommessa. Mi mancano nomi e avversari, quindi tengo una linea prudente.",
        tags=["v5", "train", "classic", "lineup_advice", "test_focus"],
    )


def test_assemble_dataset_appends_valid_p4_rows(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"

    write_jsonl(
        base_path,
        [
            example(
                mode="mantra",
                task="lineup_advice",
                source="v2_manual",
                user="Base Mantra?",
                assistant="Base answer.",
                tags=["v2", "train", "mantra"],
            ),
            example(
                mode="classic",
                task="lineup_advice",
                source="v2_manual",
                user="Base Classic?",
                assistant="Base answer.",
                tags=["v2", "train", "classic"],
            ),
        ],
    )
    write_jsonl(block_path, [valid_mantra(), valid_classic()])
    write_manifest(manifest_path, block_path)

    summary = assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)

    assert summary == {
        "base_examples": 2,
        "p4_examples": 2,
        "total_examples": 4,
        "p4_by_mode": {"mantra": 1, "classic": 1},
        "final_by_mode": {"mantra": 2, "classic": 2},
    }
    assert sum(1 for _ in output_path.open("r", encoding="utf-8")) == 4


def test_assemble_dataset_rejects_dataset_v3_or_v4_base(tmp_path: Path) -> None:
    base_path = tmp_path / "datasets" / "v3" / "train.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Base Mantra?"), valid_classic(user="Base Classic?")])
    write_jsonl(block_path, [valid_mantra(), valid_classic()])
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="must not use Dataset v3"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_duplicate_base_prompt(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Duplicate prompt?"), valid_classic(user="Base Classic?")])
    write_jsonl(block_path, [valid_mantra(user="Duplicate prompt?"), valid_classic()])
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="Duplicate user prompt"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_mantra_forbidden_terms(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Base Mantra?"), valid_classic(user="Base Classic?")])
    bad = valid_mantra()
    bad["messages"][-1]["content"] = "Sceglierei questa strada perche il modificatore protegge il reparto."
    write_jsonl(block_path, [bad, valid_classic()])
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="forbidden Mantra term"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_classic_role_code_token_but_not_letters_inside_words(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Base Mantra?"), valid_classic(user="Base Classic?")])
    clean = valid_classic()
    clean["messages"][-1]["content"] = "Sceglierei il centrocampo stabile. Mi manca il calendario, ma in una rosa corta proteggo panchina e voto medio prima del bonus."
    leaking = valid_classic(user="Classic role leak?")
    leaking["messages"][-1]["content"] = "Sceglierei il reparto sicuro, ma la W cambia il ragionamento."
    write_jsonl(block_path, [valid_mantra(), clean])
    write_manifest(manifest_path, block_path)
    assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)

    write_jsonl(block_path, [valid_mantra(), leaking])
    with pytest.raises(AssemblyError, match="forbidden Classic role code"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_broken_terms(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Base Mantra?"), valid_classic(user="Base Classic?")])
    bad = valid_classic()
    bad["messages"][-1]["content"] = "Sceglierei la punteggianza migliore se il reparto resta coperto."
    write_jsonl(block_path, [valid_mantra(), bad])
    write_manifest(manifest_path, block_path)

    with pytest.raises(AssemblyError, match="broken generated term"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)


def test_assemble_dataset_rejects_final_balance_mismatch(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    block_path = tmp_path / "drafts" / "block.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "train.jsonl"
    write_jsonl(base_path, [valid_mantra(user="Base Mantra?"), valid_classic(user="Base Classic?")])
    write_jsonl(block_path, [valid_mantra(), valid_classic()])
    write_manifest(manifest_path, block_path, final_balance={"mantra": 3, "classic": 1})

    with pytest.raises(AssemblyError, match="expected final mode split"):
        assemble_dataset(base_path=base_path, manifest_path=manifest_path, output_path=output_path)
