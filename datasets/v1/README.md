# Dataset v1

Dataset v1 is the first targeted repair set after Qwen LoRA v0.

The goal is not to make the model memorize players, votes, or current-season facts. The goal is to teach the `coachino` how FantaBrain should reason and speak:

- Mantra role codes must stay literal: `E`, `M`, `T`, `W`, `Pc`, `Dc`, `A`.
- Classic must stay at the same quality level as Mantra, with Classic-specific reasoning.
- Missing context must trigger grounded refusal, not invented facts.
- Answers should be shorter, decision-first, and conditional.

## Authoring Contract

Every row must keep the same JSONL schema used by Dataset v0:

```json
{
  "mode": "mantra",
  "task": "lineup_advice",
  "source": "v1_manual",
  "quality_score": 5,
  "tags": ["v1", "train", "mantra", "lineup_advice", "role_codes"],
  "messages": [
    {"role": "system", "content": "Sei il coach AI privato di FantaBrain..."},
    {"role": "user", "content": "Domanda utente..."},
    {"role": "assistant", "content": "Risposta ideale..."}
  ]
}
```

## Response Shape

Prefer this compact shape:

```text
Sceglierei X solo se Y. Eviterei Z se rompe A/B. In Mantra/Classic conta K. Mi mancano N dati, quindi la priorita e...
```

For refusal cases:

```text
Non posso darti un verdetto affidabile senza X. Il criterio e Y, perche Z. Mandami A/B/C e ti dico la scelta.
```

## Hard No

- No real player names.
- No made-up percentages.
- No made-up votes or scores.
- No live facts.
- No fake rules.
- No training on `benchmarks/pagella_v0.jsonl`.

## Target

See `manifest.yaml`.

Dataset v1 target is 320 examples: 160 Mantra and 160 Classic.
