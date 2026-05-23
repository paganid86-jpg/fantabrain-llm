from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantabrain_llm.dataset import load_examples, to_sft_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT/QLoRA training for FantaBrain.")
    parser.add_argument("--config", required=True, help="YAML training config.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install PyYAML or run `python -m pip install -e .[train]`.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config must be a YAML object: {path}")
    return payload


def only_supported_kwargs(target: object, values: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(target)
    return {key: value for key, value in values.items() if key in signature.parameters}


def build_model_init_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install training dependencies with `python -m pip install -e .[train]`.") from exc

    dtype_name = model_config.get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, dtype_name)

    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": model_config.get("device_map", "auto"),
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
    }

    if "low_cpu_mem_usage" in model_config:
        kwargs["low_cpu_mem_usage"] = bool(model_config["low_cpu_mem_usage"])

    token = os.getenv("HF_TOKEN")
    if token:
        kwargs["token"] = token

    if model_config.get("load_in_4bit", True):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=bool(model_config.get("use_nested_quant", True)),
        )

    return kwargs


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ModuleNotFoundError as exc:
        print("Training dependencies missing. Run: python -m pip install -e .[train]", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    lora_config = config["lora"]

    train_examples = load_examples(data_config["train_path"])
    eval_path = data_config.get("eval_path")
    eval_examples = load_examples(eval_path) if eval_path and Path(eval_path).exists() else []

    train_dataset = Dataset.from_list([to_sft_record(example) for example in train_examples])
    eval_dataset = (
        Dataset.from_list([to_sft_record(example) for example in eval_examples])
        if eval_examples
        else None
    )

    model_init_kwargs = build_model_init_kwargs(model_config)
    sft_values = {
        "output_dir": training_config["output_dir"],
        "run_name": config.get("project", {}).get("run_name"),
        "max_length": training_config.get("max_length", 4096),
        "num_train_epochs": training_config.get("num_train_epochs", 2),
        "per_device_train_batch_size": training_config.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 8),
        "learning_rate": training_config.get("learning_rate", 2e-4),
        "warmup_ratio": training_config.get("warmup_ratio", 0.03),
        "logging_steps": training_config.get("logging_steps", 5),
        "save_steps": training_config.get("save_steps", 100),
        "eval_steps": training_config.get("eval_steps", 100),
        "eval_strategy": training_config.get("eval_strategy", "steps" if eval_examples else "no"),
        "packing": training_config.get("packing", False),
        "assistant_only_loss": training_config.get("assistant_only_loss", False),
        "report_to": training_config.get("report_to", "none"),
        "seed": training_config.get("seed", 42),
        "model_init_kwargs": model_init_kwargs,
    }

    for optional_key in (
        "bf16",
        "fp16",
        "gradient_checkpointing",
        "max_grad_norm",
        "optim",
    ):
        if optional_key in training_config:
            sft_values[optional_key] = training_config[optional_key]

    if "eval_strategy" not in inspect.signature(SFTConfig).parameters:
        sft_values["evaluation_strategy"] = sft_values.pop("eval_strategy")

    sft_args = SFTConfig(**only_supported_kwargs(SFTConfig, sft_values))
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model_config["base_model"],
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(training_config["output_dir"])
    print(f"Adapter saved to {training_config['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
