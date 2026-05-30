# P2 Targeted Dataset Design

Date: 2026-05-22

Status: approved direction, pending implementation plan

## Context

FantaBrain's current candidate is `qwen25-3b-fantabrain-sft-v1`, trained from Qwen2.5-3B-Instruct with LoRA on Dataset v1. The v1 pagella used the same 40-case blind evaluation and the same decoding profile:

- `max_tokens`: 350
- `temperature`: 0.3
- `top_p`: 0.9
- `repetition_penalty`: 1.15
- `no_repeat_ngram_size`: 4

Codex manual scoring for v1 produced:

- raw average: 2.845
- effective average: 2.630
- hallucination-free: 31/40
- capped cases: 9

P2 is not a broad dataset expansion. It is a targeted repair set aimed at the patterns that capped or weakened the v1 run.

## Goal

Create Dataset v2 as Dataset v1 plus 80 hand-authored P2 examples that directly repair the worst v1 failure modes while preserving Mantra/Classic parity.

Target outcomes for the next pagella:

- effective average: 3.20 or higher
- hallucination-free: at least 36/40
- capped cases: at most 4
- no regression in Mantra/Classic balance
- answers remain short, decision-first, and grounded

## Non-Goals

- Do not train on pagella prompts or expected answers.
- Do not add real player names, current-season facts, exact future votes, invented probabilities, or fake rules.
- Do not make the model verbose to hide uncertainty.
- Do not optimize only for the existing 40-case pagella; P2 should teach reusable behavior patterns.

## Failure Map

The P2 set targets these v1 cases by failure pattern, not by copying their prompts.

| Case | Mode | Main Failure |
| --- | --- | --- |
| 2 | Mantra | Incorrect module comparison and weak `M` reasoning |
| 3 | Mantra | `P` used instead of `Pc`; fake offensive modifier logic |
| 6 | Classic | Modificatore understood only vaguely; nonsensical terms introduced |
| 9 | Mantra | Invented budget threshold and messy multirole reasoning |
| 20 | Mantra | Confusion around `C/W-A`; invented captain context |
| 27 | Mantra | Rosa modulare explained badly with irrelevant portiere/modificatore concepts |
| 28 | Classic | Modificatore explanation misses vote average and extra points |
| 34 | Classic | Risk-management answer breaks down and invents nonsense |
| 39 | Classic | Refusal starts correctly but becomes ungrounded and confused |

Secondary borderline cases:

- case 7: Classic captain reasoning lacks rigorist/matchup/regulation handling
- case 15: Classic portiere with modificatore is too evasive
- case 31: risk/varianza is unclear
- case 35: favorite-state reasoning is not crisp enough
- case 36: doubtful starter vs safe reserve needs clearer zero-risk logic

## Dataset Shape

Dataset v2 should contain Dataset v1 plus 80 new P2 examples.

P2 examples:

- total: 80
- Mantra: 40
- Classic: 40
- quality score: 5 for every row
- source: `v2_manual`
- tags must include `v2`, `train`, mode, task, and one P2 focus tag

Expected file layout:

```text
datasets/v2/
  README.md
  manifest.yaml
  drafts/
    p2_block_001_classic_modificatore.jsonl
    p2_block_002_mantra_role_codes_guardrail.jsonl
    p2_block_003_risk_varianza_decisioni.jsonl
    p2_block_004_refusal_grounded_clean.jsonl
    p2_block_005_italiano_asciutto_decision_first.jsonl
  train.jsonl
```

If `datasets/v1/train.jsonl` is not present locally, the notebook flow must restore or rebuild the final v1 dataset before assembling `datasets/v2/train.jsonl`.

## P2 Blocks

### Block 001: Classic Modificatore

File: `datasets/v2/drafts/p2_block_001_classic_modificatore.jsonl`

Examples: 20 Classic

Primary targets: cases 6, 13, 15, 28, 34

Tasks:

- lineup_advice
- auction_advice
- rules_explanation
- risk_management

Must teach:

- modificatore difesa depends on voto medio and clean defensive floor
- portiere matters as part of the defensive block, but does not replace good defenders
- a fourth reliable defender can be worth more than a medium bonus gamble
- Classic uses porta/difesa/centrocampo/attacco, not Mantra role-code logic
- avoid stray words such as modulo, slot, incastro unless they are genuinely natural in Classic context

Answer pattern:

```text
Sceglierei X se il difensore/portiere alza il voto medio. Il modificatore non e un bonus casuale: premia un reparto solido. Se invece Y ha alta probabilita di bonus e il modificatore non scatta, allora Z. Mi mancano...
```

### Block 002: Mantra Role Codes Guardrail

File: `datasets/v2/drafts/p2_block_002_mantra_role_codes_guardrail.jsonl`

Examples: 20 Mantra

Primary targets: cases 2, 3, 9, 20, 27

Tasks:

- lineup_advice
- auction_advice
- trade_advice
- rules_explanation

Must teach:

- `Pc` must never become `P`
- `M`, `E`, `T`, `W`, `A`, `Dc`, and `Pc` must be used literally
- Mantra decisions are about legal slots, modules, coverage, and which absence breaks a formation
- do not invent captains, budget thresholds, percentages, or hidden rules
- when uncertain, ask for modules and available role codes, not unrelated data

Answer pattern:

```text
Preferirei X perche copre il ruolo Y nei moduli che vuoi usare. Eviterei Z se lascia scoperto [role code]. La domanda non e solo bonus: e quale assenza ti rompe piu formazioni. Mi mancano...
```

### Block 003: Risk, Varianza, Decisioni

File: `datasets/v2/drafts/p2_block_003_risk_varianza_decisioni.jsonl`

Examples: 16 mixed

Mode split:

- Mantra: 8
- Classic: 8

Primary targets: cases 31, 34, 35, 36

Tasks:

- risk_management
- lineup_advice
- trade_advice

Must teach:

- favorite state: protect floor first, accept upside only if it does not create a clear zero-risk
- underdog state: raise ceiling with targeted risk, not with every volatile player available
- doubtful starter vs safe reserve: risk only with reliable cover or when matchup state demands upside
- first-place roster management: stability and repeatability matter more than spectacle

Answer pattern:

```text
Da favorito sceglierei X per proteggere il floor. Da sfavorito posso accettare Y, ma solo se non crea uno zero probabile. Il rischio giusto e mirato, non totale. Mi mancano...
```

### Block 004: Refusal Grounded Clean

File: `datasets/v2/drafts/p2_block_004_refusal_grounded_clean.jsonl`

Examples: 12 mixed

Mode split:

- Mantra: 6
- Classic: 6

Primary targets: cases 37, 38, 39, 40

Tasks:

- refusal_grounding
- lineup_advice
- risk_management

Must teach:

- refuse impossible certainty directly
- do not continue with invented alternatives after refusing
- provide a useful criterion after the refusal
- ask for the minimum necessary data

Required pattern:

```text
Non posso inventare X. Posso pero stimare Y usando A, B e C. Mandami questi dati e ti restituisco una scelta ordinata.
```

### Block 005: Italiano Asciutto, Decision First

File: `datasets/v2/drafts/p2_block_005_italiano_asciutto_decision_first.jsonl`

Examples: 12 mixed

Mode split:

- Mantra: 6
- Classic: 6

Primary targets: all borderline cases with acceptable reasoning but poor delivery

Tasks:

- lineup_advice
- auction_advice
- trade_advice
- rules_explanation
- risk_management

Must teach:

- start with the decision
- keep the answer between 70 and 110 words
- one main criterion per paragraph-sized answer
- no broken Italian, no filler, no invented jargon
- close with one useful missing-data request only when needed

Preferred opening markers:

- `Sceglierei`
- `Preferirei`
- `Eviterei`
- `Non posso`

## Answer Contract

Every P2 assistant answer must satisfy these rules:

- maximum 110 words
- first sentence contains the decision or refusal
- one clear tactical criterion
- one conditional branch using `solo se`, `a meno che`, or `se invece`
- no more than one missing-context sentence
- no fake numeric thresholds unless they are present in the user prompt
- no invented roles, modules, rules, votes, percentages, or future facts

## Mode Guardrails

### Mantra

Allowed core vocabulary:

- modulo
- slot
- incastro
- copertura
- ruolo
- E
- M
- T
- W
- Pc
- Dc
- A

Mantra examples must explain role-code consequences concretely:

- which role is rare
- which module becomes playable or unplayable
- whether the choice protects formation legality or only adds upside

### Classic

Allowed core vocabulary:

- porta
- difesa
- centrocampo
- attacco
- voto
- bonus
- malus
- modificatore
- panchina
- titolare
- rigorista

Classic examples must not pretend Mantra role-code constraints exist. They can discuss formation shape, but must not rely on Mantra-style role codes or arbitrary slot logic.

## Assembly And Training

Dataset v2 assembly:

- validate each P2 draft block individually
- validate total P2 count is 80
- validate P2 mode split is 40 Mantra and 40 Classic
- validate no duplicate user prompts
- append P2 examples to final Dataset v1
- write `datasets/v2/train.jsonl`

Training target:

- adapter name: `qwen25-3b-fantabrain-sft-v2`
- base model: `Qwen/Qwen2.5-3B-Instruct`
- method: QLoRA on Colab T4
- config should derive from `configs/sft/qwen25-3b-qlora-v1.yaml`
- keep the same decoding settings for evaluation

## Evaluation

After training, run the same 40-case pagella:

- eval path: `benchmarks/pagella_v0.jsonl`
- output run: `qwen25-3b-fantabrain-sft-v2-pagella-v0`
- decoding unchanged from v1
- score with the same manual rubric:
  - mode
  - tactical
  - grounded
  - clarity
  - tone
  - hallucination_free

Required comparison:

- v1 scored summary vs v2 scored summary
- Mantra average vs Classic average
- hallucination-free delta
- capped cases delta
- review of the original P2 target cases

## Acceptance Criteria

P2 is successful if the next pagella reaches:

- effective average >= 3.20
- hallucination-free count >= 36/40
- capped cases <= 4
- no severe regression in Classic compared with Mantra
- no case contains invented money, fake rules, fake votes, fake percentages, or role-code corruption

If the model improves grounding but not tactical quality, P3 should focus on richer tactical examples. If tactical quality improves but hallucinations remain high, P3 should focus on stricter refusal and uncertainty examples.
