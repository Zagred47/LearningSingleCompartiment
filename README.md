# Hay Single Compartment

Un singolo compartimento conduttanza-based, ispirato ai meccanismi del neurone
L5PC di Hay, pensato per generare dataset sequenziali piccoli e confrontare
surrogati neurali come MLP, RNN, GRU, LSTM e una ConvLSTM temporale capiente.

> **Ambito scientifico.** Questo è un modello ridotto *Hay-inspired*: conserva
> famiglie di canali e memoria dinamica interessanti, ma non è numericamente
> equivalente al modello Hay multicompartmentale originale. Non richiede
> NEURON né meccanismi NMODL compilati.

## Cosa viene simulato

Esiste una sola tensione di membrana. Il suo stato Markoviano comprende 17
variabili:

- tensione e calcio intracellulare;
- gate `NaTa_t`, `Nap_Et2`, `Kdr`, `SKv3_1`, `Im`, `Ih`, `Ca_LVAst`,
  `Ca_HVA` e `SK_E2`;
- conduttanze sinaptiche AMPA, NMDA voltage-dependent e GABA-A.

Gli input casuali combinano corrente Ornstein-Uhlenbeck, impulsi transitori e
conteggi di eventi sinaptici Poisson. Un processo a regimi alterna fasi quiete,
bilanciate, eccitatorie, inibitorie e burst. Ogni traiettoria usa un seed
distinto, e train/validation/test sono separati a livello di traiettoria.

Il file HDF5 salva, senza stato nascosto del simulatore:

- `states`: tutte le 17 variabili a ogni boundary temporale;
- `inputs`: corrente iniettata e conteggi AMPA/NMDA/GABA;
- `currents`: 13 correnti individuali più la corrente ionica totale;
- `spikes`, `regimes`, griglia temporale, seed, configurazione e nomi/ordine
  delle feature.

## Avvio rapido

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -e ".[test,notebook]"
python scripts/generate_dataset.py
python scripts/train_baselines.py artifacts/single_compartment.h5 --epochs 12
pytest
```

La configurazione riproducibile è in `configs/default.json`. Il dataset di
default contiene 12 traiettorie train, 3 validation e 3 test da 500 ms,
campionate a 0,1 ms con integrazione interna a 0,025 ms.

## Notebook Kaggle

Aprire `notebooks/kaggle_single_compartment.ipynb`, attivare una GPU Kaggle e
usare **Run all**. Il notebook:

1. installa il package locale;
2. genera e valida il dataset HDF5;
3. visualizza input, tensione, calcio e conduttanze;
4. addestra MLP, GRU, LSTM e ConvLSTM con la stessa pipeline; la ConvLSTM usa
   convoluzioni causali dilatate, circa un milione di parametri e 20 epoche;
5. confronta RMSE one-step, baseline di persistenza e rollout autoregressivo;
6. salva checkpoint e metriche in `/kaggle/working/hay_single_results`.

Per usare direttamente GitHub in Kaggle, clonare prima la repo nella directory
`/kaggle/working` e aprire/eseguire il notebook dalla root del checkout.

## API essenziale

```python
from hay_single_compartment import RandomDrive, SingleCompartmentHay

inputs, regimes = RandomDrive().sample(steps=5000, dt_ms=0.1, seed=42)
trajectory = SingleCompartmentHay().simulate(inputs, dt_ms=0.1, internal_dt_ms=0.025)
print(trajectory["states"].shape)  # (5001, 17)
```

Gli artefatti generati (`.h5`, `.pt`, cartella `artifacts/`) sono ignorati da
Git. I manifest JSON accanto ai dataset includono validazione e SHA-256.

## Struttura

```text
src/hay_single_compartment/  simulatore, protocolli, dataset, modelli, training
scripts/                      CLI di generazione e training
configs/default.json          esperimento riproducibile
notebooks/                    workflow Kaggle completo
tests/                        test numerici, schema HDF5 e modelli
```
