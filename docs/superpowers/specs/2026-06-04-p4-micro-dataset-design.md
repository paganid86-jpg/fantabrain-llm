# P4 Micro Dataset Design

## Context

Qwen LoRA v2 is still the best FantaBrain adapter baseline.

Recent scored Pagella v0 results:

| Adapter | Raw Average | Effective Average | Hallucination-Free |
|---|---:|---:|---:|
| `qwen25-3b-fantabrain-sft-v2` | 3.15 | 2.69 | 26/40 |
| `qwen25-3b-fantabrain-sft-v3` | 2.47 | 2.06 | 20/40 |
| `qwen25-3b-fantabrain-sft-v4` | 2.88 | 2.54 | 26/40 |

Dataset v3 was too broad and regressed badly. Dataset v4/P3bis reduced the v3 damage but still failed promotion gates:

- v4 did not beat v2 effective average;
- v4 did not beat v2 hallucination-free count;
- v4 raw average stayed below 3.10;
- case 2 still invented `3-5-2-0`;
- `modificatore` still appeared in Mantra cases 3, 20, 31, and 38;
- Classic answers still showed broad cross-mode vocabulary and malformed Italian.

P4 must therefore be smaller, sharper, and more guarded than P3/P3bis.

## Goal

Create a micro corrective dataset:

`datasets/v5/train.jsonl = datasets/v2/train.jsonl + 16 P4 examples`

The output adapter will be:

`models/adapters/qwen25-3b-fantabrain-sft-v5`

The adapter must be trained fresh from `Qwen/Qwen2.5-3B-Instruct` on Dataset v5. It must not continue from v2, v3, or v4 adapters.

The goal is not more domain knowledge. The goal is to test whether a tiny, clean corrective add-on can reduce lexical leakage and malformed Italian without hurting v2's overall decision quality.

## Non-Goals

- Do not use Dataset v3 or Dataset v4 rows as the training base.
- Do not train on Pagella v0 prompts or expected answers.
- Do not create a broad dataset expansion.
- Do not add real player names, live facts, future votes, odds, prices, injuries, or mutable football data.
- Do not add app serving, RAG, API integration, or production deployment work.
- Do not solve every weak tactical case in one pass.

## Dataset Shape

Dataset v5 should contain 296 total examples:

- 280 examples from Dataset v2;
- 16 new P4 examples;
- final mode balance: 148 Mantra / 148 Classic.

P4 split:

| Block | Examples | Mode Split | Focus |
|---|---:|---|---|
| `mantra_no_modificatore` | 4 | 4 Mantra / 0 Classic | Mantra answers that never mention Classic modifier language |
| `classic_clean_vocab` | 4 | 0 Mantra / 4 Classic | Classic answers using reparto/voto/bonus vocabulary only |
| `no_invented_modules` | 4 | 4 Mantra / 0 Classic | Use only modules explicitly provided in the user prompt |
| `italian_decision_clean` | 4 | 0 Mantra / 4 Classic | Short, decision-first Classic answers with clean Italian |

P4 should stay at 16 examples unless the first authoring pass cannot satisfy the balance and quality gates. In that case, the maximum allowed size is 20 examples, still balanced 50/50.

## Target Cases

Primary repair cases from v4:

- Mantra: 2, 3, 20, 31, 38
- Classic: 14, 28, 36, 39

Signals worth preserving from v4:

- case 17: Mantra protects rare-role structure well;
- case 19: Mantra correctly notices missing `M` and surplus `Pc`;
- case 37: Mantra refusal is short and grounded;
- case 34: Classic risk management improves when it uses risk/upside without overreacting.

The P4 examples should not copy these Pagella prompts. They should recreate the failure shape with different wording.

## Answer Contract

Every P4 assistant target must follow this contract:

- 45 to 75 words.
- First sentence starts with `Sceglierei`, `Preferirei`, `Eviterei`, or `Non posso`.
- The decision or refusal appears in the first sentence.
- At most one conditional branch.
- At most one missing-context sentence.
- No invented modules, formations, numeric thresholds, prices, votes, probabilities, scores, or player names.
- No malformed words, mojibake, random capitalization, or pseudo-Italian.
- No explanations framed as "do not say X" inside the assistant target. The target should simply model the clean answer.

Mode-specific contract:

- Mantra targets may use `slot`, `incastro`, `copertura`, `ruolo`, `Pc`, `T`, `W`, `M`, `C`, `E`, `Dc`, `Ds`, and `Dd` only when relevant.
- Mantra targets must not use `modificatore`, `reparto`, or Classic-only modifier reasoning.
- Classic targets may use `reparto`, `titolarita`, `bonus`, `malus`, `panchina`, `voto medio`, and `modificatore` only when relevant.
- Classic targets must not use `slot`, `incastro`, `codice`, `Pc`, `T`, `W`, `M`, `C`, `E`, `Dc`, `Ds`, or `Dd`.
- Classic targets must not mention Mantra, even as a contrast.

Role-code checks must be token-aware. For example, `M` should be rejected as a standalone role code in a Classic target, but not because it appears inside normal words.

## Prompt Contract

P4 user prompts should be natural and concise.

Allowed:

- Mantra prompts that mention provided modules, role codes, coverage, and doubts.
- Classic prompts that mention reparti, bonus, modifier rules, panchina, and risk.
- Refusal prompts asking for unknowable future/live facts.

Not allowed:

- Prompts copied from Pagella v0.
- Prompts copied from any existing training row.
- Prompts that inject forbidden words only to test whether the target avoids them, unless that word is required by the mode.
- Prompts with real players, real teams, future match facts, or mutable live data.

## Validation Gates

Dataset v5 assembly must reject P4 rows when:

- `source` is not `v5_manual`;
- `quality_score` is below 5;
- required tags are missing;
- a P4 user prompt duplicates a Pagella v0 prompt;
- a P4 user prompt duplicates any existing Dataset v0/v1/v2 prompt;
- a P4 user prompt duplicates another P4 prompt;
- a Mantra target contains `modificatore` or `reparto`;
- a Classic target contains `slot`, `incastro`, `codice`, `Mantra`, or token-aware Mantra role-code matches;
- any target contains known malformed words from the v4 audit, including `offENSIVO`, `sicurata`, `multicolore`, `punteggianza`, `malusso`, `maleducata`, `aspettarello`, or `esattissimi`;
- total P4 count is not 16;
- P4 mode split is not 8 Mantra / 8 Classic;
- final Dataset v5 mode split is not 148 Mantra / 148 Classic.

Validation should be strict. P4 is a leakage-control experiment, so rejecting a risky row is better than teaching the unwanted wording again.

## Training Design

Add a new config:

`configs/sft/qwen25-3b-qlora-v5.yaml`

It should mirror the stable v2/v4 Colab-friendly QLoRA setup:

- train path: `datasets/v5/train.jsonl`;
- output dir: `models/adapters/qwen25-3b-fantabrain-sft-v5`;
- base model: `Qwen/Qwen2.5-3B-Instruct`;
- 4-bit loading enabled;
- `torch_dtype: float16`;
- `bf16: false`;
- `fp16: false`;
- `low_cpu_mem_usage: true`;
- same established Pagella decoding defaults.

The first run should keep two epochs for comparability. If v5 fails, the next experiment should not add more data immediately; first compare examples and training dynamics against v2/v4.

## Evaluation And Promotion Rule

Evaluate v5 on the unchanged blind Pagella v0.

Promote v5 over v2 only if all gates pass:

- effective average is greater than v2's 2.69;
- hallucination-free count is greater than 26/40;
- raw average is at least 3.10;
- case 2 does not invent any module;
- Mantra outputs contain zero `modificatore`;
- Classic outputs contain no Mantra role-code leakage;
- cases 14, 28, 36, and 39 improve or at least do not regress versus v2;
- no new malformed Italian pattern appears in more than one case.

If v5 fails these gates, keep v2 as the best adapter baseline.

## Manual Notebook Flow

The notebook should remain a runner and learning surface.

Expected flow:

1. Clone or restore the repo from `master` plus the P4 branch.
2. Upload or restore `fantabrain-dataset-v2-280.zip`.
3. Author the four P4 draft blocks under `datasets/v5/drafts/`.
4. Run Dataset v5 assembly and strict audit.
5. Download `fantabrain-dataset-v5-296.zip`.
6. Train `qwen25-3b-fantabrain-sft-v5` fresh from base Qwen.
7. Download adapter zip before evaluation.
8. Evaluate on Pagella v0 with the established decoding defaults.
9. Download Pagella v5 zip.
10. Score v5 against v2 before deciding promotion.

## Risks

- Sixteen rows may be too weak to change the adapter behavior.
- If P4 targets overcorrect, they may reduce useful tactical flexibility.
- If targets include forbidden words as examples of what not to say, SFT may reinforce them.
- v5 may tie v2 on hallucination-free count but still lose raw clarity, as v4 did.
- Strict vocabulary gates can reject semantically reasonable rows, but this is acceptable for a micro leakage experiment.

## Decision

Proceed with P4 as a 16-example micro corrective dataset, balanced 8 Mantra / 8 Classic, assembled as Dataset v5 from Dataset v2 only. Train a fresh Qwen LoRA v5 from base. Keep v2 as rollback and promote v5 only if it beats v2 on effective score and hallucination-free count while passing all leakage and malformed-Italian gates.
