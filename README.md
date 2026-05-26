# FantaBrain LLM

Palestra professionale per addestrare e valutare un modello dedicato alla AI Chat di FantaBrain.

Obiettivo: creare un `coachino` basato su un modello instruct open source, partendo da Llama 3.1 8B Instruct, allenato con dataset nostri su Fantacalcio Mantra e Classic.

## Principi

- Il fine-tuning insegna stile, formato, ragionamento e comportamento da coach.
- Dati mutevoli come voti, rose, calendario e statistiche devono arrivare dal contesto applicativo o da RAG, non essere memorizzati nel modello.
- Ogni esperimento deve essere ripetibile: dataset, config, adapter e report restano tracciati.
- Il notebook serve solo come runner; la logica vive nel repo.

## Struttura

```text
fantabrain-llm/
  configs/sft/              # Config training QLoRA/SFT
  data/raw/                 # Dataset grezzo locale, non versionato
  data/processed/           # Dataset pronto per training, non versionato
  data/eval/                # Pagelle/eval set, non versionate
  docs/                     # Design, piani, note operative
  examples/raw/             # Mini dataset versionato per smoke test
  models/adapters/          # Adapter LoRA, non versionati
  notebooks/                # Notebook runner, non logica primaria
  reports/runs/             # Output eval e report run, non versionati
  scripts/                  # CLI per prepare, train, eval, merge
  src/fantabrain_llm/       # Core schema e dataset utilities
  tests/                    # Test leggeri senza GPU
```

## Setup locale

```powershell
cd fantabrain-llm
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Se `py` non e disponibile, installa Python 3.11+ o usa il runtime del tuo ambiente cloud.

## Primo giro dati

```powershell
python scripts/prepare_dataset.py `
  --input examples/raw/seed_conversations.jsonl `
  --output data/processed/seed_train.jsonl `
  --eval-output data/eval/seed_eval.jsonl `
  --eval-ratio 0.34
```

Dataset v0:

```powershell
python scripts/prepare_dataset.py `
  --input datasets/v0/train.jsonl `
  --output data/processed/v0_train.jsonl `
  --eval-output data/eval/v0_holdout.jsonl `
  --eval-ratio 0
```

## Smoke test

```powershell
python -m pytest
```

## Training su GPU Linux

Questo passaggio e pensato per Colab, Kaggle, RunPod o Modal, non per Windows locale.

```bash
python -m pip install -e ".[train]"
python scripts/train_lora.py --config configs/sft/llama31-8b-qlora.yaml
```

Il primo target e `meta-llama/Llama-3.1-8B-Instruct` con QLoRA.

Forgia Qwen v0 su Colab T4:

```bash
python -m pip install -e ".[train]"
python -m pip install -U "bitsandbytes>=0.46.1"
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v0.yaml
```

Il runbook operativo e `docs/runbooks/qwen25-lora-v0.md`.

Forgia Qwen v1, solo dopo aver completato `datasets/v1/train.jsonl`:

```bash
python -m pip install -e ".[train]"
python -m pip install -U "bitsandbytes>=0.46.1"
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v1.yaml
```

Forgia Qwen v2, solo dopo aver completato `datasets/v2/train.jsonl`:

```bash
python -m pip install -e ".[train]"
python -m pip install -U "bitsandbytes>=0.46.1"
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v2.yaml
```

Il runbook operativo e `docs/runbooks/qwen25-lora-v2.md`.

Forgia Qwen v3, solo dopo aver completato `datasets/v3/train.jsonl`:

```bash
python -m pip install -e ".[train]"
python -m pip install -U "bitsandbytes>=0.46.1"
python scripts/train_lora.py --config configs/sft/qwen25-3b-qlora-v3.yaml
```

Il runbook operativo e `docs/runbooks/qwen25-lora-v3.md`.

## Pagella manuale

```powershell
python scripts/run_eval.py `
  --eval data/eval/seed_eval.jsonl `
  --run-name seed-manual-review
```

Pagella v0:

```powershell
python scripts/run_eval.py `
  --eval benchmarks/pagella_v0.jsonl `
  --run-name pagella-v0-manual-review
```

La pagella crea un report Markdown con prompt, risposta attesa e checklist di qualita.

## Baseline predictions

Smoke locale senza modello reale:

```powershell
python scripts/generate_predictions.py `
  --provider echo `
  --model echo-baseline `
  --eval benchmarks/pagella_v0.jsonl `
  --run-name echo-pagella-v0-smoke
```

Baseline su GPU con Transformers:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name base-llama31-pagella-v0
```

Pagella con adapter Qwen v0:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v0 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v0-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

Pagella con adapter Qwen v2:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v2 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v2-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

Pagella con adapter Qwen v3:

```bash
python scripts/generate_predictions.py \
  --provider transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter models/adapters/qwen25-3b-fantabrain-sft-v3 \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name qwen25-3b-fantabrain-sft-v3-pagella-v0 \
  --load-in-4bit \
  --torch-dtype float16 \
  --max-tokens 350 \
  --temperature 0.3 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4
```

Baseline su endpoint OpenAI-compatible:

```bash
python scripts/generate_predictions.py \
  --provider openai-compatible \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --eval benchmarks/pagella_v0.jsonl \
  --run-name base-api-pagella-v0
```

Gli output finiscono in `reports/runs/<run-name>/` come `predictions.jsonl`, `comparison.md` e `summary.json`.

## P1 Scoring

Per trasformare una pagella generata in metriche aggregate:

```powershell
python scripts/create_scores_template.py `
  --predictions reports/runs/<run-name>/predictions.jsonl
```

Compila `scores.template.csv`, salvalo come `scores.csv`, poi lancia:

```powershell
python scripts/score_predictions.py `
  --predictions reports/runs/<run-name>/predictions.jsonl `
  --scores reports/runs/<run-name>/scores.csv
```

Il CSV deve usare queste colonne:

```csv
case,mode,tactical,grounded,clarity,tone,hallucination_free,notes
```

`hallucination_free` e binario: `1` se la risposta non inventa dati/regole/nomi, `0` se allucina. Quando vale `0`, il punteggio effettivo del case viene plafonato a `1.0`.

Dataset v1 e descritto in `datasets/v1/manifest.yaml`.
Dataset v2 e descritto in `datasets/v2/manifest.yaml` e nel runbook `docs/runbooks/qwen25-lora-v2.md`.
Dataset v3 e descritto in `datasets/v3/manifest.yaml` e nel runbook `docs/runbooks/qwen25-lora-v3.md`.

## Slang

- `palestra`: questo repo e il workflow training.
- `forgia`: una run di training.
- `coachino`: un adapter/modello candidato.
- `pagella`: eval set e report di valutazione.
