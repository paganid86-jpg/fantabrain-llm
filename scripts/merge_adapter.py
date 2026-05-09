from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base-model", required=True, help="Base model id or local path.")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path.")
    parser.add_argument("--output", required=True, help="Merged model output path.")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        print("Install training dependencies with: python -m pip install -e .[train]")
        print(str(exc))
        return 1

    dtype = getattr(torch, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(output)
    print(f"Merged model written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
