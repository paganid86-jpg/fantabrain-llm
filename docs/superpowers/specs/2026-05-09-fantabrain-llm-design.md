# FantaBrain LLM Palestra Design

## Goal

Create a separate professional training repository named `fantabrain-llm` for building a FantaBrain AI Chat model through supervised fine-tuning and QLoRA practice.

## Scope

The first version covers:

- Dataset schema for Fantacalcio Mantra and Classic examples.
- Local validation and deterministic train/eval split.
- A seed dataset for smoke testing.
- A Llama 3.1 8B Instruct QLoRA config.
- Training, adapter merge, and manual evaluation scripts.

The first version does not deploy into FantaBrain yet. Integration with `/api/ai/chat` happens after a candidate `coachino` passes the pagella.

## Architecture

The repo separates lightweight local work from GPU work. Schema validation, dataset preparation, and manual eval run without GPU. Training and merge scripts depend on Hugging Face, PEFT, TRL, and a Linux GPU environment such as Colab, Kaggle, RunPod, or Modal.

The notebook layer is intentionally thin. Notebooks launch scripts and inspect outputs, while reusable logic lives in `src/fantabrain_llm` and `scripts`.

## Dataset Format

Each JSONL row contains:

- `mode`: `mantra` or `classic`.
- `task`: a small task name such as `lineup_advice`, `auction_advice`, or `trade_advice`.
- `messages`: chat-style turns beginning with `system` and ending with `assistant`.
- optional `source`, `quality_score`, and `tags`.

This makes mode differences explicit so the model learns when Mantra-specific role logic matters and when Classic advice should stay reparto/bonus focused.

## Training Direction

The first base model is `meta-llama/Llama-3.1-8B-Instruct` with QLoRA. The model choice keeps the experiment small enough for learning while preserving a possible future path toward Groq-compatible Llama 3.1 adapter inference.

## Evaluation

The first pagella is manual and repeatable. It renders the same eval cases for every candidate model and scores:

- mode correctness;
- tactical usefulness;
- groundedness;
- concise reasoning;
- FantaBrain coach tone.

Automated model inference eval can be added after the first adapter exists.
