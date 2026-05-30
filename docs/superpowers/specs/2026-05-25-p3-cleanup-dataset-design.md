# P3 Cleanup Dataset Design

Date: 2026-05-25

Status: approved direction, pending implementation plan

## Context

FantaBrain's current candidate is `qwen25-3b-fantabrain-sft-v2`, trained from Qwen2.5-3B-Instruct with LoRA on Dataset v2.

The v2 pagella used the same 40-case blind benchmark and the established decoding profile:

- `max_tokens`: 350
- `temperature`: 0.3
- `top_p`: 0.9
- `repetition_penalty`: 1.15
- `no_repeat_ngram_size`: 4

Codex manual scoring for v2 produced:

- raw average: 3.150
- effective average: 2.690
- hallucination-free: 26/40
- capped cases: 14

Compared with v1, P2 improved raw decision quality but reduced hallucination-free reliability. P3 is therefore not a broad knowledge expansion. It is a small cleanup pass aimed at grounding, mode separation, and language hygiene.

## Goal

Create Dataset v3 as Dataset v2 plus 40 hand-authored P3 cleanup examples.

P3 should reduce error patterns that make otherwise useful answers unsafe or confusing:

- invented Mantra modules
- Mantra and Classic concepts mixed together
- weak Classic `modificatore difesa` explanations
- refusal answers that start correctly but continue with invented specifics
- broken Italian, malformed role terms, and artificial jargon

Target outcomes for the next pagella:

- hallucination-free: at least 32/40
- effective average: 2.90 or higher
- no Mantra case with invented modules not present in the prompt
- Classic modificatore cases average at least 3.5/5 on the relevant rubric dimensions
- no severe loss in raw tactical quality compared with v2

## Non-Goals

- Do not add another broad tactical dataset.
- Do not train on pagella prompts or expected answers.
- Do not optimize only for the existing 40-case pagella.
- Do not add live player facts, current-season facts, exact future votes, fake probabilities, fake prices, or fake rules.
- Do not make answers longer to sound safer.
- Do not introduce a new base model or change the decoding profile.

## Failure Map

The P3 set targets these v2 cases by reusable failure pattern, not by copying their prompts.

| Case | Mode | Main Failure |
| --- | --- | --- |
| 2 | Mantra | Invented modules `3-5-2` and `4-5-1`; missed given comparison |
| 5 | Classic | Brought in off-context concepts such as modules and trade logic |
| 6 | Classic | Misexplained modificatore with invented opponent/module logic |
| 7 | Classic | Captaincy reasoning missed rigorist, centrality, matchup, regulation |
| 9 | Mantra | Correct direction but messy language and stray modificatore reference |
| 10 | Mantra | Correct direction but still mixes bonus/voto/modificatori loosely |
| 19 | Mantra | Major role-balance error: says `M` is not needed if many `Pc` exist |
| 25 | Mantra | Double-role explanation still too generic and mentions modificatori |
| 27 | Mantra | Rosa modulare explanation derails into budget, codice, rigori |
| 28 | Classic | Modificatore explanation misses vote-average and defensive-block core |
| 29 | Classic | Equilibrio tra reparti confused with rare-slot logic |
| 32 | Mantra | Ballottaggi answer derails into market/scambio language |
| 34 | Classic | Underdog risk logic is inconsistent and partly inverted |
| 37 | Mantra | Refusal starts correctly, then pollutes answer with unrelated options |

Secondary cleanup cases:

- case 15: portiere plus modificatore is too vague
- case 31: underdog risk in Mantra is contradictory
- case 36: doubtful starter vs safe reserve needs clearer cover logic
- case 40: future-vote refusal is correct but too imprecise

## Dataset Shape

Dataset v3 should contain Dataset v2 plus 40 new P3 examples.

P3 examples:

- total: 40
- Mantra: 20
- Classic: 20
- quality score: 5 for every row
- source: `v3_manual`
- tags must include `v3`, `train`, mode, task, and one P3 focus tag

Expected file layout:

```text
datasets/v3/
  README.md
  manifest.yaml
  drafts/
    p3_block_001_mantra_no_module_invention.jsonl
    p3_block_002_classic_modificatore_clean.jsonl
    p3_block_003_refusal_stop_clean.jsonl
    p3_block_004_mantra_roles_no_cross_mode.jsonl
    p3_block_005_italiano_cleanup_decision_first.jsonl
  train.jsonl
```

If `datasets/v2/train.jsonl` is not present locally or in Colab, the notebook flow must restore it from `fantabrain-dataset-v2-280.zip` before assembling `datasets/v3/train.jsonl`.

## P3 Blocks

### Block 001: Mantra No Module Invention

File: `datasets/v3/drafts/p3_block_001_mantra_no_module_invention.jsonl`

Examples: 8 Mantra

Primary targets: cases 2, 3, 31, 32

Tasks:

- lineup_advice
- risk_management
- rules_explanation

Must teach:

- only compare modules explicitly named by the user
- never introduce a new module as if it were available
- if module legality is unclear, ask for available role codes instead of inventing an option
- preserve role codes exactly as written: `Pc` must not become `P`
- when a module has a weak `M`, `E`, `W`, or `Pc`, explain the concrete slot consequence

Answer pattern:

```text
Sceglierei tra i moduli che mi hai dato, non ne aggiungerei un terzo. X ha senso se [slot/role condition]. Y diventa migliore solo se [condition]. Mi mancano i ruoli disponibili per chiuderla.
```

### Block 002: Classic Modificatore Clean

File: `datasets/v3/drafts/p3_block_002_classic_modificatore_clean.jsonl`

Examples: 10 Classic

Primary targets: cases 6, 13, 15, 28, 29

Tasks:

- lineup_advice
- auction_advice
- rules_explanation

Must teach:

- modificatore difesa is about defensive vote average and extra points
- portiere matters as part of the defensive block
- reliable defenders can be worth more than low-probability bonus gambles
- Classic reasoning uses reparti, not Mantra slot constraints
- do not mention Mantra role codes, rare slots, or invented module mechanics in Classic answers

Answer pattern:

```text
Sceglierei X se alza il voto medio della difesa. Il modificatore premia un reparto solido, non un nome casuale. Se invece Y ha bonus molto piu probabile e il modificatore non scatta, allora puoi cambiare scelta.
```

### Block 003: Refusal Stop Clean

File: `datasets/v3/drafts/p3_block_003_refusal_stop_clean.jsonl`

Examples: 8 mixed

Mode split:

- Mantra: 4
- Classic: 4

Primary targets: cases 37, 38, 39, 40

Tasks:

- refusal_grounding
- lineup_advice
- risk_management

Must teach:

- refuse impossible certainty in the first sentence
- after refusing, provide only a criterion and a minimal data request
- do not invent candidates, modules, rankings, vote projections, match facts, or role availability
- do not use unrelated examples after the refusal

Required pattern:

```text
Non posso sapere X senza dati aggiornati o senza la tua rosa. Posso pero aiutarti a stimare Y usando A, B e C. Mandami questi dati e ti restituisco una scelta ordinata.
```

### Block 004: Mantra Roles No Cross-Mode

File: `datasets/v3/drafts/p3_block_004_mantra_roles_no_cross_mode.jsonl`

Examples: 8 Mantra

Primary targets: cases 9, 10, 19, 20, 25, 27

Tasks:

- auction_advice
- trade_advice
- rules_explanation

Must teach:

- do not use Classic concepts such as modificatore difesa when the prompt is Mantra and the user did not ask for them
- value `E`, `M`, `Pc`, `T`, `W/A`, `C`, and `Dc` by formation usability, not by generic bonus talk
- surplus in one Mantra role can still be structurally bad if another required role is missing
- a double-role or multirole player is valuable because it opens and protects modules
- a role-safe reserve can beat a flashier attacker if it prevents broken formations

Answer pattern:

```text
Preferirei X perche protegge [role code] nei moduli che vuoi usare. Y porta piu upside, ma non compensa se lascia scoperto [role code]. In Mantra il valore nasce dall'incastro, non solo dal bonus.
```

### Block 005: Italiano Cleanup Decision First

File: `datasets/v3/drafts/p3_block_005_italiano_cleanup_decision_first.jsonl`

Examples: 6 Classic

Primary targets: all cases with acceptable direction but broken delivery

Tasks:

- lineup_advice
- trade_advice
- risk_management
- rules_explanation

Must teach:

- first sentence is a decision or refusal
- answer length should stay between 55 and 95 words
- use plain Italian football/fantasy language
- no malformed words such as `multiruoco`, `sicurata`, `offENSIVO`, `assettogliando`
- no artificial jargon such as `codice`, `rotore`, or `punteggiatura` unless the user uses those words
- no more than one missing-data sentence

Preferred opening markers:

- `Sceglierei`
- `Preferirei`
- `Eviterei`
- `Non posso`

## Answer Contract

Every P3 assistant answer must satisfy these rules:

- maximum 95 words
- first sentence contains the decision or refusal
- use only entities, roles, modules, and facts present in the user prompt unless asking for missing data
- one clear tactical criterion
- no more than one conditional branch
- no more than one missing-context sentence
- no fake numeric thresholds unless they are present in the user prompt
- no invented roles, modules, rules, votes, percentages, player names, match facts, or future facts
- no cross-mode vocabulary unless the prompt explicitly asks to compare modes

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
- C

Mantra examples must explain:

- which role is structurally scarce
- which named module becomes safer or weaker
- whether the choice protects formation legality or only adds upside
- why a surplus role may still be less useful than a missing role

Mantra examples must not use Classic-specific concepts such as `modificatore difesa` unless the user explicitly mentions a hybrid rule.

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
- capitano

Classic examples must explain:

- reparto balance
- vote floor
- bonus upside
- panchina and cover
- modifier effect when relevant

Classic examples must not rely on Mantra role-code constraints or rare-slot logic.

## Assembly And Training

Dataset v3 assembly:

- validate each P3 draft block individually
- validate total P3 count is 40
- validate P3 mode split is 20 Mantra and 20 Classic
- validate all P3 examples have quality score 5
- validate no duplicate user prompts
- validate no exact pagella prompt leakage
- append P3 examples to final Dataset v2
- write `datasets/v3/train.jsonl`

Training target:

- adapter name: `qwen25-3b-fantabrain-sft-v3`
- base model: `Qwen/Qwen2.5-3B-Instruct`
- method: QLoRA on Colab T4
- config should derive from `configs/sft/qwen25-3b-qlora-v2.yaml`
- keep the same decoding settings for evaluation

## Evaluation

After training, run the same 40-case pagella:

- eval path: `benchmarks/pagella_v0.jsonl`
- output run: `qwen25-3b-fantabrain-sft-v3-pagella-v0`
- decoding unchanged from v2
- score with the same manual rubric:
  - mode
  - tactical
  - grounded
  - clarity
  - tone
  - hallucination_free

Required comparison:

- v2 scored summary vs v3 scored summary
- hallucination-free delta
- effective average delta
- specific review of v2 regressed cases
- check for invented module names in Mantra cases
- check for Classic/Mantra vocabulary leakage

## Acceptance Criteria

P3 is successful if the next pagella reaches:

- hallucination-free count >= 32/40
- effective average >= 2.90
- raw average does not fall below 3.05
- zero Mantra answers invent a module not present in the prompt
- zero Classic answers use Mantra role-code logic unless the prompt explicitly asks for it
- Classic modificatore-related cases average at least 3.5/5 across tactical, grounded, and clarity
- no answer contains broken generated words that obscure meaning

If hallucination-free improves but tactical quality drops, P4 should add richer but still grounded tactical examples. If tactical quality stays high but hallucinations remain above 8 capped cases, P4 should focus only on refusal and uncertainty discipline.
