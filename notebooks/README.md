# Notebooks

I notebook qui sono runner, non la fonte della logica.

Regola pratica:

1. Clona o apri questa repo in Colab, Kaggle o RunPod.
2. Installa il pacchetto con `python -m pip install -e ".[train]"`.
3. Lancia gli script in `scripts/`.
4. Salva adapter, report e config della forgia.

La logica di dataset, training ed eval deve restare in `src/` e `scripts/`.
