# Baseline Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable prediction harness that runs Pagella v0 through a base model and writes comparable outputs.

**Architecture:** Provider-specific inference lives behind a small chat client interface. Prediction/report generation is provider-agnostic and locally testable through an `echo` provider.

**Tech Stack:** Python 3.11 stdlib for local tests, optional Transformers/PyTorch for GPU execution, optional OpenAI-compatible HTTP endpoint through stdlib `urllib`.

---

### Task 1: Prediction Report Core

**Files:**
- Create: `tests/test_predictions.py`
- Create: `src/fantabrain_llm/predictions.py`

- [x] Write failing tests for generating prediction records and report files.
- [x] Run the tests and verify they fail because `fantabrain_llm.predictions` does not exist.
- [x] Implement prediction record creation, JSONL writing, Markdown comparison rendering, and summary writing.
- [x] Run targeted tests and verify they pass.

### Task 2: Provider Clients

**Files:**
- Create: `tests/test_inference.py`
- Create: `src/fantabrain_llm/inference.py`

- [x] Write failing tests for the `echo` provider and provider factory validation.
- [x] Run the tests and verify they fail because `fantabrain_llm.inference` does not exist.
- [x] Implement `ChatClient`, `EchoChatClient`, `TransformersChatClient`, `OpenAICompatibleChatClient`, and `make_chat_client`.
- [x] Run targeted tests and verify they pass.

### Task 3: CLI Harness

**Files:**
- Create: `scripts/generate_predictions.py`
- Create: `tests/test_generate_predictions_cli.py`

- [x] Write a failing CLI smoke test using `--provider echo`.
- [x] Run the test and verify it fails because the script does not exist.
- [x] Implement the CLI wrapper around dataset loading, client creation, and report writing.
- [x] Run targeted CLI test and verify it passes.

### Task 4: Docs And Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-05-10-baseline-eval-harness.md`

- [x] Add README commands for echo, transformers, and OpenAI-compatible baseline runs.
- [x] Run `python scripts/generate_predictions.py --provider echo --model echo-baseline --eval benchmarks/pagella_v0.jsonl --run-name echo-pagella-v0-smoke`.
- [x] Run `python -m compileall src scripts tests`.
- [x] Run `python -m pytest`.
- [x] Mark this plan complete and commit the branch.
