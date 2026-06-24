# OpenAI Fallback Eval v0 Design

## Context

`qwen25-3b-fantabrain-sft-v2` remains the current best local adapter, but it is not safe enough to be the only app-chat brain. The merged output filter v0 now gives the repo a deterministic way to decide whether an answer can be shown, should trigger a fallback, or should be replaced by a conservative safe response.

The next step should be eval-first. Before wiring FantaBrain app production traffic to a paid provider, the palestra should measure how often Qwen/LoRA needs fallback and whether `gpt-5.4-mini` repairs those cases cleanly.

Official OpenAI docs currently position `gpt-5.4-mini` as a lower-latency, lower-cost model variant, available through the Responses API and SDKs. Current standard pricing is $0.75 input / $4.50 output per 1M tokens. See:

- https://developers.openai.com/api/docs/models
- https://developers.openai.com/api/docs/pricing
- https://developers.openai.com/api/docs/guides/text?api-mode=responses

## Goal

Add an offline fallback evaluation path that takes an existing Qwen/LoRA `predictions.jsonl`, runs the output filter on every primary answer, calls `gpt-5.4-mini` only for blocked cases, filters the fallback answer too, and writes a comparison report.

The experiment should answer four product questions:

- How often does the current primary model trigger fallback?
- Which hard checks trigger fallback most often?
- Does `gpt-5.4-mini` clear those failures after a second filter pass?
- What is the rough fallback token/cost footprint before app integration?

## Non-Goals

- Do not integrate with the FantaBrain app yet.
- Do not train a new adapter.
- Do not modify Pagella v0 or training datasets.
- Do not store OpenAI API keys, tokens, secrets, or `.env` values.
- Do not make `gpt-5.5` the default fallback.
- Do not add streaming, retries, user accounts, billing controls, or production telemetry yet.
- Do not post-process model text to hide failed reasoning.

## Design Summary

The first implementation should work over existing prediction runs:

```text
primary predictions.jsonl
  -> filter primary prediction
  -> if pass/pass_with_warnings: keep primary answer
  -> if fallback/safe: call gpt-5.4-mini with original prompt
  -> filter fallback answer
  -> if pass/pass_with_warnings: keep fallback answer
  -> if fallback/safe: use conservative safe response
  -> write fallback eval reports
```

This branch is a bridge between offline eval and future app integration. It should produce the same decision fields the app will eventually need, but it should not touch the app repo.

## Fallback Trigger Rules

The fallback should run when the primary `FilterDecision.action` is:

- `fallback`
- `safe`

The fallback should not run when the primary action is:

- `pass`
- `pass_with_warnings`

This keeps costs low and avoids replacing acceptable Qwen/LoRA answers unnecessarily.

## Fallback Prompt Shape

The fallback request should avoid feeding the bad primary answer back into GPT in v0. The fallback gets only:

- a compact FantaBrain system instruction;
- the mode (`mantra` or `classic`);
- the task;
- the original user prompt from `predictions.jsonl`.

Recommended system instruction:

```text
Sei il coach AI privato di FantaBrain.
Rispondi in italiano pulito, con decisione prima e spiegazione breve.
Rispetta la modalita richiesta: Mantra o Classic.
Non inventare moduli, nomi giocatori, voti futuri, probabilita o dati live non forniti.
Se manca contesto, dichiaralo e dai solo un criterio prudente.
```

Mode-specific rules:

```text
Mantra: ragiona con ruoli, slot, coperture, moduli citati e panchina compatibile. Non usare modificatore o reparto.
Classic: ragiona con reparti, titolarita, bonus/malus, modificatore difesa se pertinente e panchina per reparto. Non usare codici ruolo Mantra se non citati.
```

The fallback answer should stay short. Initial default:

- model: `gpt-5.4-mini`
- max output tokens: 350
- temperature: 0.2
- no web search
- no tools

## OpenAI Client Shape

Add a dedicated client for official OpenAI fallback calls rather than overloading the existing `openai-compatible` provider. The existing provider targets `/chat/completions` style compatible endpoints. For official OpenAI fallback, use the Responses API.

Proposed module:

- `src/fantabrain_llm/openai_fallback.py`

Proposed API:

```python
@dataclass(frozen=True)
class FallbackUsage:
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None


@dataclass(frozen=True)
class FallbackResponse:
    text: str
    model: str
    usage: FallbackUsage


class OpenAIFallbackClient:
    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,
        max_output_tokens: int = 350,
        temperature: float = 0.2,
    ) -> None: ...

    def generate(self, *, mode: str, task: str, prompt: str) -> FallbackResponse: ...
```

The client should read `OPENAI_API_KEY` from the environment when `api_key` is not passed. Missing key should raise a clear local error. Tests should mock HTTP calls and must not call the real API.

Implementation can use stdlib `urllib.request` to avoid adding a runtime dependency, unless a later implementation plan explicitly chooses the official OpenAI SDK.

## Report Shape

Add a CLI that evaluates fallback behavior over a prediction run:

```bash
python scripts/run_fallback_eval.py \
  --predictions reports/runs/<primary-run>/predictions.jsonl \
  --output-dir reports/runs/<primary-run>-fallback-eval-v0 \
  --fallback-model gpt-5.4-mini \
  --max-output-tokens 350 \
  --temperature 0.2
```

Required outputs:

- `fallback_eval.json`
- `fallback_eval.md`
- `fallback_predictions.jsonl`

Each case result should include:

- `case_id`
- `mode`
- `task`
- `primary_prediction`
- `primary_action`
- `primary_reason`
- `primary_violations`
- `fallback_used`
- `fallback_prediction`
- `fallback_action`
- `fallback_reason`
- `fallback_violations`
- `final_prediction`
- `final_source`: `primary`, `fallback`, or `safe`
- `usage`
- `estimated_cost_usd`

Summary should include:

- total cases
- primary action counts
- fallback used count
- fallback success count
- final source counts
- unresolved safe count
- violation counts before and after fallback
- estimated total fallback cost

## Safe Response

If the fallback answer also filters to `fallback` or `safe`, use a conservative final answer instead of either model output.

Initial safe response:

```text
Damn, non ho abbastanza contesto per chiuderla con sicurezza.
Ti do una lettura prudente: evita scelte basate su dati non confermati e confronta titolarita, copertura, modalita e rischio prima di decidere.
```

The safe response should be explicit in reports with `final_source: "safe"`.

## Cost Controls

The implementation should be conservative by default:

- fallback only on `fallback` or `safe`;
- max output tokens default 350;
- no tools or web search;
- no retry loop in v0;
- all API usage visible in the report;
- rough local cost estimate based on configured model price constants;
- no hidden calls during tests.

Cost estimate constants for `gpt-5.4-mini` standard pricing:

- input: 0.75 USD / 1M tokens
- output: 4.50 USD / 1M tokens

If usage tokens are missing from the API response, report `estimated_cost_usd: null` rather than guessing from characters.

## Error Handling

Missing `OPENAI_API_KEY`:

- fail fast with a clear message;
- do not silently mark every case as safe.

OpenAI request failure:

- fail the CLI by default;
- future app integration may use safe response on provider outage, but offline eval should expose failures loudly.

Malformed OpenAI response:

- fail the CLI with a clear provider error.

Filter failure:

- fail the CLI because deterministic filtering is part of the tested pipeline.

## Testing

Use TDD. Required tests:

- fallback is not called for primary `pass`;
- fallback is not called for primary `pass_with_warnings`;
- fallback is called for primary `fallback`;
- fallback is called for primary `safe`;
- fallback `pass` becomes `final_source: fallback`;
- fallback hard failure becomes `final_source: safe`;
- batch report preserves case ids and violations;
- missing API key raises a clear error;
- mocked Responses API payload extracts text and usage;
- CLI writes all three outputs with mocked client;
- CLI reports provider errors cleanly without traceback.

Tests must not call the real OpenAI API.

## Colab / Local Usage Direction

This feature is mostly local/API based, not GPU based. It can run on a normal machine or Colab CPU runtime because it consumes existing prediction files and calls OpenAI only for fallback cases.

Required user setup:

- set `OPENAI_API_KEY` as an environment variable or Colab secret;
- do not commit the key;
- run against an existing Qwen/LoRA prediction package.

## Promotion Gate

The fallback experiment is promising if:

- fallback is used on a minority of cases;
- most fallback cases become `pass` or `pass_with_warnings`;
- unresolved `safe` cases are rare;
- estimated cost is acceptable for expected app traffic;
- manual review confirms fallback answers are not generic or misleading.

If fallback usage is high, that is a signal that Qwen/LoRA is not ready as primary for public app traffic.

## Future App Integration

If this eval succeeds, the next branch should target the FantaBrain app backend contract:

```text
AI chat request
  -> primary model service
  -> output filter
  -> optional OpenAI fallback
  -> second output filter
  -> final response + metadata log
```

App integration must keep provider keys server-side only. The frontend should never call OpenAI directly.

## Rollback

This is eval-only. If results are poor or costs look too high, do not integrate fallback into the app. Keep output filter v0 and continue improving the primary model or try a different fallback strategy.
