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
    """Raised when Dataset v5 cannot be assembled safely."""


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


def contains_phrase(text: str, term: str) -> bool:
    return term.lower() in text.lower()


def contains_role_code(text: str, role_code: str) -> bool:
    return re.search(rf"(?<![A-Za-z]){re.escape(role_code)}(?![A-Za-z])", text) is not None


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
    classic_role_codes = [str(term) for term in gates.get("classic_forbidden_role_codes", [])]
    broken_terms = [str(term) for term in gates.get("broken_terms", [])]

    for term in broken_terms:
        if term in text:
            raise AssemblyError(f"{prefix}: broken generated term {term!r}")

    if mode == "mantra":
        for term in mantra_terms:
            if contains_phrase(text, term):
                raise AssemblyError(f"{prefix}: forbidden Mantra term {term!r}")

    if mode == "classic":
        for term in classic_terms:
            if contains_phrase(text, term):
                raise AssemblyError(f"{prefix}: forbidden Classic term {term!r}")
        for role_code in classic_role_codes:
            if contains_role_code(text, role_code):
                raise AssemblyError(f"{prefix}: forbidden Classic role code {role_code!r}")


def validate_p4_record(
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

    required_tags = {"v5", "train", str(mode), str(task), focus_tag}
    missing_tags = sorted(required_tags - {str(tag) for tag in tags})
    if missing_tags:
        raise AssemblyError(f"{prefix}: missing tags: {', '.join(missing_tags)}")

    prompt = normalize_prompt(user_prompt(record))
    if prompt in forbidden_prompts:
        raise AssemblyError(f"{prefix}: P4 prompt matches a pagella prompt")

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
    normalized = raw_path.replace("\\", "/")
    if normalized.startswith(("datasets/", "benchmarks/")):
        return ROOT / candidate
    return manifest_path.parent / candidate


def count_modes(records: list[dict[str, object]], expected_modes: dict[str, int]) -> dict[str, int]:
    counter = Counter(str(record["mode"]) for record in records)
    return {mode: counter.get(mode, 0) for mode in expected_modes}


def validate_base_path(base_path: Path, quality_gates: dict[str, object]) -> None:
    normalized_base = str(base_path).replace("\\", "/")
    forbid_v3 = bool(quality_gates.get("forbid_dataset_v3_as_base", False))
    forbid_v4 = bool(quality_gates.get("forbid_dataset_v4_as_base", False))
    if forbid_v3 and "datasets/v3/" in normalized_base:
        raise AssemblyError("Dataset v5 must not use Dataset v3 as base")
    if forbid_v4 and "datasets/v4/" in normalized_base:
        raise AssemblyError("Dataset v5 must not use Dataset v4 as base")


def assemble_dataset(base_path: Path, manifest_path: Path, output_path: Path) -> dict[str, object]:
    manifest = load_yaml(manifest_path)
    quality_gates = manifest.get("quality_gates", {})
    validate_base_path(base_path, quality_gates)

    base_records = records_from_examples(base_path)
    eval_path = ROOT / manifest.get("blind_eval_path", "benchmarks/pagella_v0.jsonl")
    forbidden_prompts = load_pagella_prompts(
        eval_path,
        required=bool(quality_gates.get("forbid_eval_prompt_leakage", False)),
    )

    vocabulary_gates = manifest.get("mode_vocabulary_gates", {})
    min_quality = int(quality_gates.get("min_quality_score", 5))
    required_source = str(quality_gates.get("source", "v5_manual"))

    p4_records: list[dict[str, object]] = []
    for block in manifest.get("p4_blocks", []):
        block_path = resolve_manifest_path(manifest_path, str(block["path"]))
        expected_examples = int(block["examples"])
        expected_split = block["mode_split"]
        focus_tag = str(block["focus_tag"])

        records = records_from_examples(block_path)
        if len(records) != expected_examples:
            raise AssemblyError(
                f"{block_path}: expected {expected_examples} examples, got {len(records)}"
            )

        split = count_modes(records, expected_split)
        if split != expected_split:
            raise AssemblyError(f"{block_path}: expected mode split {expected_split}, got {split}")

        for index, record in enumerate(records, start=1):
            validate_p4_record(
                record=record,
                path=block_path,
                index=index,
                min_quality=min_quality,
                required_source=required_source,
                focus_tag=focus_tag,
                forbidden_prompts=forbidden_prompts,
                vocabulary_gates=vocabulary_gates,
            )

        p4_records.extend(records)

    expected_total = int(manifest["p4_examples"])
    if len(p4_records) != expected_total:
        raise AssemblyError(f"expected {expected_total} P4 examples, got {len(p4_records)}")

    expected_p4_by_mode = manifest["p4_balance"]["by_mode"]
    p4_by_mode = count_modes(p4_records, expected_p4_by_mode)
    if p4_by_mode != expected_p4_by_mode:
        raise AssemblyError(f"expected P4 mode split {expected_p4_by_mode}, got {p4_by_mode}")

    output_records = [*base_records, *p4_records]
    validate_unique_prompts(output_records)

    expected_final_by_mode = manifest["final_balance"]["by_mode"]
    final_by_mode = count_modes(output_records, expected_final_by_mode)
    if final_by_mode != expected_final_by_mode:
        raise AssemblyError(
            f"expected final mode split {expected_final_by_mode}, got {final_by_mode}"
        )

    write_jsonl(output_path, output_records)

    return {
        "base_examples": len(base_records),
        "p4_examples": len(p4_records),
        "total_examples": len(output_records),
        "p4_by_mode": p4_by_mode,
        "final_by_mode": final_by_mode,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble and validate Dataset v5.")
    parser.add_argument("--base", required=True, help="Final Dataset v2 JSONL path.")
    parser.add_argument("--manifest", required=True, help="Dataset v5 manifest path.")
    parser.add_argument("--output", required=True, help="Output Dataset v5 train JSONL path.")
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

    print("Dataset v5 assembled")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
