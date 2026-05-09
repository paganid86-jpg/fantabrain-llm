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

## Slang

- `palestra`: questo repo e il workflow training.
- `forgia`: una run di training.
- `coachino`: un adapter/modello candidato.
- `pagella`: eval set e report di valutazione.
