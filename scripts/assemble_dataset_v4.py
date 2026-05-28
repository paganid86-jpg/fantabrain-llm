from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.dataset import (  # noqa: E402
    DatasetError,
    load_examples,
    to_sft_record,
    write_jsonl,
)


class AssemblyError(ValueError):
    """Raised when Dataset v4 cannot be assembled safely."""


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise AssemblyError("Install PyYAML or run `python -m pip install -e .[dev]`.") from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise AssemblyError(f"Manifest must be a YAML object: {path}")
    return payload


def user_prompt(record: dict[str, object]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise AssemblyError("messages must be a list")
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise AssemblyError("example must include a user prompt")


def assistant_text(record: dict[str, object]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise AssemblyError("messages must be a list")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise AssemblyError("example must include an assistant target")


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def records_from_examples(path: Path) -> list[dict[str, object]]:
    try:
        return [to_sft_record(example) for example in load_examples(path)]
    except DatasetError as exc:
        raise AssemblyError(str(exc)) from exc


def load_pagella_prompts(eval_path: Path, *, required: bool) -> set[str]:
    if not eval_path.exists():
        if required:
            raise AssemblyError(f"Pagella eval file not found: {eval_path}")
        return set()
    return {normalize_prompt(user_prompt(record)) for record in records_from_examples(eval_path)}


def contains_forbidden_term(text: str, term: str) -> bool:
    lowered = text.lower()
    escaped = re.escape(term.lower())
    if term.lower() in {"pc", "dc", "ds", "dd"}:
        return re.search(rf"(?<![a-z]){escaped}(?![a-z])", lowered) is not None
    return term.lower() in lowered


def contains_broken_term(text: str, term: str) -> bool:
    return term in text


def validate_target_vocabulary(
    *,
    mode: str,
    text: str,
    path: Path,
    index: int,
    gates: dict[str, object],
) -> None:
    prefix = f"{path}:{index}"
    mantra_terms = [str(term) for term in gates.get("mantra_forbidden_terms", [])]
    classic_terms = [str(term) for term in gates.get("classic_forbidden_terms", [])]
    broken_terms = [str(term) for term in gates.get("broken_terms", [])]

    for term in broken_terms:
        if contains_broken_term(text, term):
            raise AssemblyError(f"{prefix}: broken generated term {term!r}")

    if mode == "mantra":
        for term in mantra_terms:
            if contains_forbidden_term(text, term):
                raise AssemblyError(f"{prefix}: forbidden Mantra term {term!r}")

    if mode == "classic":
        for term in classic_terms:
            if contains_forbidden_term(text, term):
                raise AssemblyError(f"{prefix}: forbidden Classic term {term!r}")


def validate_p3bis_record(
    *,
    record: dict[str, object],
    path: Path,
    index: int,
    min_quality: int,
    required_source: str,
    focus_tag: str,
    forbidden_prompts: set[str],
    vocabulary_gates: dict[str, object],
) -> None:
    prefix = f"{path}:{index}"
    if record.get("source") != required_source:
        raise AssemblyError(f"{prefix}: source must be {required_source}")
    quality_score = record.get("quality_score")
    if not isinstance(quality_score, int) or quality_score < min_quality:
        raise AssemblyError(f"{prefix}: quality_score must be at least {min_quality}")

    mode = record.get("mode")
    task = record.get("task")
    tags = record.get("tags")
    if mode not in {"mantra", "classic"}:
        raise AssemblyError(f"{prefix}: invalid mode {mode!r}")
    if not isinstance(task, str) or not task:
        raise AssemblyError(f"{prefix}: task is required")
    if not isinstance(tags, list):
        raise AssemblyError(f"{prefix}: tags must be a list")

    required_tags = {"v4", "train", str(mode), str(task), focus_tag}
    missing_tags = sorted(required_tags - {str(tag) for tag in tags})
    if missing_tags:
        raise AssemblyError(f"{prefix}: missing tags: {', '.join(missing_tags)}")

    prompt = normalize_prompt(user_prompt(record))
    if prompt in forbidden_prompts:
        raise AssemblyError(f"{prefix}: P3bis prompt matches a pagella prompt")

    validate_target_vocabulary(
        mode=str(mode),
        text=assistant_text(record),
        path=path,
        index=index,
        gates=vocabulary_gates,
    )


def validate_unique_prompts(records: list[dict[str, object]]) -> None:
    seen: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        prompt = normalize_prompt(user_prompt(record))
        if prompt in seen:
            raise AssemblyError(f"Duplicate user prompt at rows {seen[prompt]} and {index}")
        seen[prompt] = index


def resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    root_candidate = ROOT / candidate
    if root_candidate.exists():
        return root_candidate
    return manifest_path.parent / candidate


def count_modes(records: list[dict[str, object]], expected_modes: dict[str, int]) -> dict[str, int]:
    counter = Counter(str(record["mode"]) for record in records)
    return {mode: counter.get(mode, 0) for mode in expected_modes}


def assemble_dataset(base_path: Path, manifest_path: Path, output_path: Path) -> dict[str, object]:
    manifest = load_yaml(manifest_path)
    quality_gates = manifest.get("quality_gates", {})
    if bool(quality_gates.get("forbid_dataset_v3_as_base", False)):
        normalized_base = str(base_path).replace("\\", "/")
        if "datasets/v3/" in normalized_base:
            raise AssemblyError("Dataset v4 must not use Dataset v3 as base")

    base_records = records_from_examples(base_path)
    eval_path = ROOT / manifest.get("blind_eval_path", "benchmarks/pagella_v0.jsonl")

    vocabulary_gates = manifest.get("mode_vocabulary_gates", {})
    min_quality = int(quality_gates.get("min_quality_score", 5))
    required_source = str(quality_gates.get("source", "v4_manual"))
    forbidden_prompts = load_pagella_prompts(
        eval_path,
        required=bool(quality_gates.get("forbid_eval_prompt_leakage", False)),
    )

    p3bis_records: list[dict[str, object]] = []
    for block in manifest.get("p3bis_blocks", []):
        block_path = resolve_manifest_path(manifest_path, str(block["path"]))
        expected_examples = int(block["examples"])
        expected_split = block["mode_split"]
        focus_tag = str(block["focus_tag"])

        records = records_from_examples(block_path)
        if len(records) != expected_examples:
            raise AssemblyError(f"{block_path}: expected {expected_examples} examples, got {len(records)}")

        split = count_modes(records, expected_split)
        if split != expected_split:
            raise AssemblyError(f"{block_path}: expected mode split {expected_split}, got {split}")

        for index, record in enumerate(records, start=1):
            validate_p3bis_record(
                record=record,
                path=block_path,
                index=index,
                min_quality=min_quality,
                required_source=required_source,
                focus_tag=focus_tag,
                forbidden_prompts=forbidden_prompts,
                vocabulary_gates=vocabulary_gates,
            )

        p3bis_records.extend(records)

    expected_total = int(manifest["p3bis_examples"])
    if len(p3bis_records) != expected_total:
        raise AssemblyError(f"expected {expected_total} P3bis examples, got {len(p3bis_records)}")

    expected_by_mode = manifest["p3bis_balance"]["by_mode"]
    p3bis_by_mode = count_modes(p3bis_records, expected_by_mode)
    if p3bis_by_mode != expected_by_mode:
        raise AssemblyError(f"expected P3bis mode split {expected_by_mode}, got {p3bis_by_mode}")

    output_records = [*base_records, *p3bis_records]
    validate_unique_prompts(output_records)
    write_jsonl(output_path, output_records)

    return {
        "base_examples": len(base_records),
        "p3bis_examples": len(p3bis_records),
        "total_examples": len(output_records),
        "p3bis_by_mode": p3bis_by_mode,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble and validate Dataset v4.")
    parser.add_argument("--base", required=True, help="Final Dataset v2 JSONL path.")
    parser.add_argument("--manifest", required=True, help="Dataset v4 manifest path.")
    parser.add_argument("--output", required=True, help="Output Dataset v4 train JSONL path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = assemble_dataset(
            base_path=Path(args.base),
            manifest_path=Path(args.manifest),
            output_path=Path(args.output),
        )
    except AssemblyError as exc:
        print(f"Assembly error: {exc}", file=sys.stderr)
        return 1

    print("Dataset v4 assembled")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
