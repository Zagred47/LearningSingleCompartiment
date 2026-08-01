# Hay Single Compartment

Un singolo compartimento conduttanza-based, ispirato ai meccanismi del neurone
L5PC di Hay, pensato per generare dataset sequenziali piccoli e confrontare
surrogati neurali come MLP, RNN, GRU, LSTM e una ConvLSTM temporale capiente.

> **Ambito scientifico.** Questo è un modello ridotto *Hay-inspired*: conserva
> famiglie di canali e memoria dinamica interessanti, ma non è numericamente
> equivalente al modello Hay multicompartmentale originale. Non richiede
> NEURON né meccanismi NMODL compilati.

La repo include anche `FaithfulHaySoma`, un secondo simulatore separato che
trascrive le cinetiche e i parametri somatici originali di
`L5PCbiophys3.hoc` (Hay 2011, Figura 4). È fedele al singolo compartimento
somatico, ma non pretende di riprodurre il cavo multicompartmentale completo.

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

Per il compartimento fedele usare
`notebooks/kaggle_faithful_hay_soma.ipynb`. Genera traiettorie da 2 s con tutti
i 17 stati intrinseci del soma Hay più tre conduttanze sinaptiche, garantisce
la copertura dei cinque regimi, mostra progress/ETA e riusa una cache HDF5 da
`/kaggle/input` o `/kaggle/working`. Il primo benchmark riparte soltanto dalla
ConvLSTM Large; il dataset può essere pubblicato come Kaggle Dataset e montato
nelle sessioni successive senza rigenerarlo.

Aprire `notebooks/kaggle_single_compartment.ipynb`, attivare una GPU Kaggle e
usare **Run all**. Il notebook:

1. installa il package locale;
2. genera e valida il dataset HDF5;
3. visualizza input, tensione, calcio e conduttanze;
4. addestra MLP, GRU, LSTM e ConvLSTM con la stessa pipeline; la ConvLSTM usa
   convoluzioni causali dilatate, circa un milione di parametri e 20 epoche;
5. confronta RMSE one-step, baseline di persistenza e rollout autoregressivo;
6. salva checkpoint e metriche in `/kaggle/working/hay_single_results`.

Per spingere la ConvLSTM, usare invece
`notebooks/kaggle_convlstm_scaling.ipynb`: genera 32 traiettorie più lunghe e
addestra soltanto ConvLSTM Large (2,2 milioni di parametri) con AMP, cosine
schedule, early stopping e progress/ETA per generazione, training e rollout.

Il primo esperimento di bias induttivo ontologico è in
`notebooks/kaggle_ontology_experiment_01.ipynb`. Confronta, a budget di
parametri quasi uguale, una GRU globale e un mosaico composto esclusivamente
da GRU standard. Nel mosaico ciascuna GRU vede soltanto le dipendenze causali
del proprio sottosistema. Il notebook misura anche l'efficienza dei dati al
25%, 50% e 100%, gli errori per entità e i rollout fino a 500 ms.

La progressione compositiva che conserva il backbone vincente parte da
`notebooks/kaggle_composite_experiment_02.ipynb`. Confronta ConvLSTM Large,
una ConvLSTM monolitica con capacità aggiuntiva e ConvLSTM Large con tre GRU
locali standard dedicate esclusivamente agli stati AMPA, NMDA e GABA. Il
confronto stabilisce se la separabilità dei recettori aggiunge informazione
strutturale oltre al semplice aumento dei parametri.

Il passo successivo è `notebooks/kaggle_composite_experiment_03_hcn.ipynb`:
mantiene il composito vincente con le tre GRU recettoriali e verifica, senza
cambiare altro, se la coordinata di stato 8 può essere assegnata a una quarta
GRU locale. Il controllo parte dallo stesso composito e aggiunge una quantità
quasi identica di parametri alla testa globale.

L'esperimento diagnostico
`notebooks/kaggle_composite_experiment_03b_auxiliary.ipynb` distingue la
mancata separabilità della coordinata 8 dalla perdita di supervisione
condivisa. Mantiene l'output locale e aggiunge al backbone una testa ausiliaria
standard, usata solo durante il training, confrontandola con un controllo di
capacità quasi identico.

`notebooks/kaggle_composite_experiment_03c_markov_mlp.ipynb` verifica poi se
l'instabilità della coordinata 8 dipende dallo stato nascosto della GRU
privata. A parità di backbone, auxiliary head e capacità, sostituisce soltanto
la GRU locale con una MLP residuale standard priva di memoria autonoma.

Per usare direttamente GitHub in Kaggle, clonare prima la repo nella directory
`/kaggle/working` e aprire/eseguire il notebook dalla root del checkout.

## Micro-neurone Hay spaziale a quattro compartimenti

Il nuovo teacher fisiologico ridotto è `basale -- soma -- tronco apicale --
tuft`. Conserva canali Hay specifici per regione, correnti assiali, calcio,
gate completi e stati a doppia esponenziale di AMPA/NMDA/GABA-A. Non possiede
un input di corrente iniettata: gli unici ingressi sono spike binari su
sinapsi eccitatorie e inibitorie fisse, allocate fra basale, tronco e tuft in
proporzione alla lunghezza dendritica rappresentata.

```bash
python scripts/generate_micro_dataset.py artifacts/hay_micro_4c_v1.h5 --workers 4
```

Su Kaggle usare `notebooks/kaggle_micro_hay_4comp_dataset.ipynb`. Il notebook
mostra avanzamento ed ETA, riusa la cache HDF5, controlla i nove regimi e
visualizza raster, potenziali, calcio e variabili lente. Stati e correnti sono
target/diagnostica e non vengono forniti come input al futuro surrogate
input-only.

Il primo training input-only è in
`notebooks/kaggle_micro_input_only_cfc_01.ipynb`. Confronta una GRU standard
con il layer CfC ufficiale di `ncps`, selezionando automaticamente una GRU con
budget di parametri vicino. Cinque bin da 0,1 ms vengono concatenati in ordine
per ogni passo di rete: il training GPU è più pratico senza perdere identità o
timing degli spike. Il burn-in è costituito esclusivamente da spike; i suoi
stati fisici sono usati soltanto come target intermedi per addestrare la
dinamica causale con truncated BPTT.
I dataset schema `1.0.0` già generati vengono migrati automaticamente al
formato `1.2.0`: il notebook ricostruisce deterministicamente soltanto i 2 s
di burn-in e pretende un replay bit-identico degli input utili prima di
accettare il file.

Per diagnosticare la GRU addestrata usare
`notebooks/kaggle_micro_gru_diagnostics_01.ipynb`: espone encoder, hidden,
decoder e gate reset/update/candidate tramite replay numericamente verificato,
separando le distribuzioni per spike, burst, rapid-fire e plateau di tuft.

Il passo event-aware completo è
`notebooks/kaggle_micro_event_aware_training_02.ipynb`. Genera e conserva un
pool più ampio di traiettorie sinaptiche fisiologiche, cataloga la risposta
effettiva del teacher e combina un passaggio naturale completo con finestre
stratificate dotate di 2 s di contesto spike-only. Confronta GRU-MSE,
GRU event-aware, Branch ELM, ConvGRU e ConvLSTM a circa 318 mila parametri.
Il test naturale resta intatto; metriche separate descrivono tutte le classi
di evento. Il contratto dettagliato è in `docs/event_aware_micro_experiment.md`.

Dopo l'ablation negativa della loss event-aware da zero, usare
`notebooks/kaggle_micro_spike_finetune_03.ipynb`. Il notebook carica la
GRU-MSE convergente e applica un fine-tuning conservativo: la MSE sui 61 stati
resta sempre attiva, mentre deficit di picco, derivata, gate rapidi e logit
spike bilanciati entrano gradualmente. Salva `last` e `best` a ogni epoca e
riprende automaticamente dopo un'interruzione Kaggle.

Per non confondere il fine-tuning con la capacità di apprendere da zero,
`notebooks/kaggle_micro_conservative_scratch_04.ipynb` addestra invece GRU,
Branch ELM, ConvGRU e ConvLSTM da inizializzazione casuale. Tutte ricevono 10
epoche di sola MSE, poi la stessa loss conservativa entra con curriculum. Il
notebook richiede l'HDF5 già montato e si arresta se non lo trova: non avvia
silenziosamente una nuova generazione.

Dopo l'esito negativo della supervisione ausiliaria di fase, usare
`notebooks/kaggle_micro_residual_tcn_finetune_06.ipynb`. La GRU-MSE viene
congelata e un adattatore TCN causale a dilatazioni `(1, 2, 4, 8)` apprende un
residuo sui 61 stati da hidden GRU e spike input. La proiezione finale parte da
zero, quindi l'epoca 0 coincide esattamente con il baseline; dataset e checkpoint
esistenti vengono scoperti automaticamente negli input Kaggle.

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
