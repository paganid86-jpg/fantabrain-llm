# FantaBrain LLM Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the `fantabrain-llm` training palestra as a separate, repeatable repository.

**Architecture:** Keep validation and eval lightweight so they run locally without GPU. Keep training and adapter merge scripts isolated behind optional GPU dependencies.

**Tech Stack:** Python 3.11+, stdlib validation utilities, pytest for local smoke tests, Hugging Face Transformers/TRL/PEFT for GPU training.

---

### Task 1: Repository Skeleton

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: tracked `.gitkeep` files under data/model/report folders

- [x] **Step 1: Create the folder layout**

Create directories for `configs`, `data`, `docs`, `examples`, `models`, `notebooks`, `reports`, `scripts`, `src`, and `tests`.

- [x] **Step 2: Add package metadata**

Add `pyproject.toml` with core, dev, and train dependency groups.

- [x] **Step 3: Add README**

Document goals, slang, setup, data preparation, smoke tests, training, and manual pagella commands.

### Task 2: Dataset Core

**Files:**
- Create: `src/fantabrain_llm/schema.py`
- Create: `src/fantabrain_llm/dataset.py`
- Create: `src/fantabrain_llm/prompts.py`
- Create: `tests/test_dataset.py`

- [x] **Step 1: Implement schema validation**

Validate `mode`, `task`, chat message roles, first system turn, last assistant turn, optional tags, and optional quality score.

- [x] **Step 2: Implement JSONL utilities**

Load JSONL with line-numbered errors, filter by quality, split deterministically, and write processed SFT records.

- [x] **Step 3: Add tests**

Cover seed dataset validity, deterministic split, invalid line errors, and SFT record preservation.

### Task 3: CLI Scripts

**Files:**
- Create: `scripts/prepare_dataset.py`
- Create: `scripts/run_eval.py`
- Create: `scripts/train_lora.py`
- Create: `scripts/merge_adapter.py`

- [x] **Step 1: Add dataset preparation CLI**

Read raw JSONL, validate it, split into train/eval, and write processed files.

- [x] **Step 2: Add manual eval CLI**

Render eval rows into a Markdown pagella with a stable rubric.

- [x] **Step 3: Add QLoRA training CLI**

Load config, create Hugging Face datasets, configure 4-bit loading, apply LoRA config, train, and save adapter output.

- [x] **Step 4: Add adapter merge CLI**

Load base model and adapter, merge LoRA weights, and save a merged local model.

### Task 4: Seed Data And Config

**Files:**
- Create: `examples/raw/seed_conversations.jsonl`
- Create: `configs/sft/llama31-8b-qlora.yaml`

- [x] **Step 1: Add seed conversations**

Include Mantra and Classic examples covering lineup, auction, trade, rules, and post-match review tasks.

- [x] **Step 2: Add first training config**

Point to Llama 3.1 8B Instruct with QLoRA defaults suitable for a first small GPU run.

### Task 5: Verification

**Files:**
- No new files expected.

- [ ] **Step 1: Run compile check**

Run: `python -m compileall src scripts tests`

Expected: all Python files compile.

- [ ] **Step 2: Run tests**

Run: `python -m pytest`

Expected: dataset tests pass after installing dev dependencies.
