# P3bis Corrective Dataset Design

## Context

Qwen LoRA v2 is the current best FantaBrain adapter baseline. Qwen LoRA v3 loaded and evaluated correctly, but regressed on Pagella v0:

- v2 raw average: 3.15
- v3 raw average: 2.47
- v2 effective average: 2.69
- v3 effective average: 2.06
- v2 hallucination-free: 26/40
- v3 hallucination-free: 20/40

The surgical audit found that v3 introduced new cross-mode leakage and new capped cases. The main failure patterns are:

- `modificatore` appearing in Mantra answers;
- `slot`, `codice`, `modulo`, or Mantra role-code language appearing in Classic answers;
- invented Mantra modules instead of staying inside the modules supplied by the prompt;
- decision inversions on favored/underdog, hype-vs-stable, and end-auction coverage cases;
- refusal answers that start correctly but continue with speculative advice;
- noisier Italian and malformed words.

Dataset v3 is therefore not a promotion candidate and must not be used as the base for the next model.

## Goal

Create Dataset v4 as a small corrective pass:

`datasets/v4/train.jsonl = datasets/v2/train.jsonl + 20 P3bis examples`

The output adapter will be `qwen25-3b-fantabrain-sft-v4`. It should be trained fresh from the base Qwen model on Dataset v4, not continued from the v3 adapter.

The goal is not broad domain expansion. The goal is to recover v2 quality while reducing the specific v2/v3 failure modes found by the audit.

## Non-Goals

- Do not include Dataset v3 rows in Dataset v4.
- Do not train on Pagella v0 prompts or expected answers.
- Do not add real player names, live facts, future votes, odds, prices, injuries, or mutable football data.
- Do not broaden into a large 40+ example pass.
- Do not solve production serving, RAG, or app integration in this phase.

## Dataset Shape

Dataset v4 should contain 300 total examples:

- 280 examples from Dataset v2;
- 20 new P3bis examples;
- final mode balance: 150 Mantra / 150 Classic.

P3bis split:

| Block | Examples | Mode Split | Focus |
|---|---:|---|---|
| `mantra_anti_leak` | 5 | 5 Mantra / 0 Classic | No `modificatore`, no invented modules, clean role/slot reasoning |
| `classic_anti_leak` | 5 | 0 Mantra / 5 Classic | No `slot`, no `codice`, no Mantra role-code language |
| `decision_inversion` | 6 | 3 Mantra / 3 Classic | Correct first-sentence decisions for cases like 4, 16, 24 |
| `refusal_stop` | 4 | 2 Mantra / 2 Classic | Refuse unknowable facts, ask for minimum data, stop |

## Target Cases

Primary repair cases:

- Mantra: 2, 3, 4, 10, 11, 20, 25, 27, 38
- Classic: 13, 15, 16, 21, 24, 28, 29, 30, 36, 40

Recovered signals to preserve:

- case 6: Classic modificatore lineup reasoning improved;
- case 9: Mantra multirole coverage improved;
- case 32: Mantra ballottaggi improved;
- case 34: Classic risk management improved;
- case 37: refusal direction improved.

The P3bis examples should imitate the recovered signals only where they are clean. They should not copy Pagella prompts.

## Answer Contract

Every P3bis assistant target must follow this contract:

- 55 to 90 words.
- First sentence starts with `Sceglierei`, `Preferirei`, `Eviterei`, or `Non posso`.
- One clear decision or refusal before any explanation.
- At most one conditional branch.
- At most one missing-context sentence.
- No invented modules; if modules are present, use only the modules named by the user prompt.
- No invented numeric votes, percentages, thresholds, prices, or scores.
- No malformed words such as `offENSIVO`, `sicurata`, `ruoloni`, or mojibake.

Mode-specific contract:

- Mantra targets may use `slot`, `incastro`, `copertura`, `ruolo`, `Pc`, `T`, `W`, `M`, `C`, `E`, `Dc`, `Ds`, `Dd` only when relevant.
- Mantra targets must not use `modificatore`.
- Classic targets may use `reparto`, `titolarita`, `bonus`, `malus`, `panchina`, `voto medio`, and `modificatore` only when relevant.
- Classic targets must not use `slot`, `codice`, `Pc`, `T`, `W`, or other Mantra role-code language.
- Classic targets must not mention Mantra to explain what not to do. The target should simply answer in Classic vocabulary.

## Validation Gates

The Dataset v4 assembly must reject P3bis rows when:

- `source` is not `v4_manual`;
- `quality_score` is below 5;
- required tags are missing;
- a P3bis user prompt duplicates a Pagella v0 prompt;
- a P3bis user prompt duplicates any existing train prompt;
- a Mantra target contains `modificatore`;
- a Classic target contains `slot`, `codice`, `modulo`, `moduli`, `Mantra`, or Mantra role-code tokens;
- a target contains known malformed words;
- total P3bis count is not 20;
- P3bis mode split is not 10 Mantra / 10 Classic.

The validation should be strict even if it rejects a row that sounds reasonable to a human. The purpose is to avoid teaching the unwanted vocabulary again.

## Training Design

Add a new config:

`configs/sft/qwen25-3b-qlora-v4.yaml`

It should mirror the stable v2/v3 Colab-friendly QLoRA setup, with:

- train path: `datasets/v4/train.jsonl`;
- output dir: `models/adapters/qwen25-3b-fantabrain-sft-v4`;
- base model: `Qwen/Qwen2.5-3B-Instruct`;
- 4-bit loading enabled;
- `torch_dtype: float16`;
- `bf16: false`;
- `fp16: false`;
- same decoding defaults for evaluation as P0/P2/P3.

The first training run should keep two epochs for comparability. If v4 still regresses, the next experiment should change only one factor at a time, starting with learning rate or epochs.

## Evaluation And Promotion Rule

Evaluate v4 on the unchanged blind Pagella v0.

Promote v4 over v2 only if:

- effective average is greater than v2's 2.69;
- hallucination-free count is greater than 26/40;
- raw average does not fall below 3.10;
- case 2 does not invent modules;
- Mantra cases do not use `modificatore`;
- Classic cases do not use Mantra vocabulary.

If v4 fails these gates, keep v2 as the best adapter baseline and use v4 only as a learning artifact.

## Manual Notebook Flow

The notebook should remain a runner and learning surface, not the source of reusable logic.

Expected flow:

1. Clone or restore the repo.
2. Upload or restore `fantabrain-dataset-v2-280.zip`.
3. Author the four P3bis draft blocks under `datasets/v4/drafts/`.
4. Run Dataset v4 assembly and strict audit.
5. Download `fantabrain-dataset-v4-300.zip`.
6. Train `qwen25-3b-fantabrain-sft-v4`.
7. Download adapter zip before evaluation.
8. Evaluate on Pagella v0 with the established decoding defaults.
9. Download Pagella v4 zip.
10. Score v4 against v2 before deciding promotion.

## Risks

- A tiny 20-row pass may be too weak to correct entrenched vocabulary.
- A strict vocabulary gate may reject rows that are semantically acceptable but risky for SFT.
- Training from base on v2 + P3bis may still regress due to dataset style, not only content.
- Reusing Pagella-like prompts would contaminate the blind benchmark, so prompt-leak validation must remain mandatory.

## Decision

Proceed with P3bis as a 20-example corrective dataset, balanced 10 Mantra / 10 Classic, assembled as Dataset v4 from Dataset v2 only. Do not use Dataset v3 as training input. Use strict mode-vocabulary gates and promote v4 only if it beats v2 on effective score and hallucination-free count without losing raw quality.
