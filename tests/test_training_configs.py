from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_qwen25_lora_v0_config_trains_on_dataset_not_pagella() -> None:
    config_path = ROOT / "configs" / "sft" / "qwen25-3b-qlora-v0.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["data"]["train_path"] == "datasets/v0/train.jsonl"
    assert config["data"].get("eval_path") in (None, "")
    assert "pagella" not in config["data"]["train_path"]
    assert config["training"]["output_dir"] == "models/adapters/qwen25-3b-fantabrain-sft-v0"
    assert config["training"]["num_train_epochs"] == 3
    assert config["model"]["load_in_4bit"] is True
    assert config["model"]["torch_dtype"] == "float16"


def test_qwen25_lora_v1_config_points_to_dataset_v1() -> None:
    config_path = ROOT / "configs" / "sft" / "qwen25-3b-qlora-v1.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["data"]["train_path"] == "datasets/v1/train.jsonl"
    assert config["data"].get("eval_path") in (None, "")
    assert "pagella" not in config["data"]["train_path"]
    assert config["training"]["output_dir"] == "models/adapters/qwen25-3b-fantabrain-sft-v1"
    assert config["training"]["num_train_epochs"] == 2
    assert config["model"]["load_in_4bit"] is True


def test_qwen25_lora_v2_config_points_to_dataset_v2() -> None:
    config_path = ROOT / "configs" / "sft" / "qwen25-3b-qlora-v2.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["data"]["train_path"] == "datasets/v2/train.jsonl"
    assert config["data"].get("eval_path") in (None, "")
    assert "pagella" not in config["data"]["train_path"]
    assert config["training"]["output_dir"] == "models/adapters/qwen25-3b-fantabrain-sft-v2"
    assert config["training"]["num_train_epochs"] == 2
    assert config["model"]["load_in_4bit"] is True
    assert config["model"]["torch_dtype"] == "float16"
    assert config["model"]["low_cpu_mem_usage"] is True
    assert config["training"]["bf16"] is False
    assert config["training"]["fp16"] is False


def test_qwen25_lora_v3_config_points_to_dataset_v3() -> None:
    config_path = ROOT / "configs" / "sft" / "qwen25-3b-qlora-v3.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["data"]["train_path"] == "datasets/v3/train.jsonl"
    assert config["data"].get("eval_path") in (None, "")
    assert "pagella" not in config["data"]["train_path"]
    assert config["training"]["output_dir"] == "models/adapters/qwen25-3b-fantabrain-sft-v3"
    assert config["training"]["num_train_epochs"] == 2
    assert config["model"]["load_in_4bit"] is True
    assert config["model"]["torch_dtype"] == "float16"
    assert config["model"]["low_cpu_mem_usage"] is True
    assert config["training"]["bf16"] is False
    assert config["training"]["fp16"] is False


def test_qwen25_lora_v4_config_points_to_dataset_v4() -> None:
    config_path = ROOT / "configs" / "sft" / "qwen25-3b-qlora-v4.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["data"]["train_path"] == "datasets/v4/train.jsonl"
    assert config["data"].get("eval_path") in (None, "")
    assert "pagella" not in config["data"]["train_path"]
    assert "datasets/v3" not in config["data"]["train_path"]
    assert config["training"]["output_dir"] == "models/adapters/qwen25-3b-fantabrain-sft-v4"
    assert config["training"]["num_train_epochs"] == 2
    assert config["model"]["load_in_4bit"] is True
    assert config["model"]["torch_dtype"] == "float16"
    assert config["model"]["low_cpu_mem_usage"] is True
    assert config["training"]["bf16"] is False
    assert config["training"]["fp16"] is False
