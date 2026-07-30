# Schedario degli stati e delle scale temporali del modello Hay

## Ambito

Questo documento descrive il modello originale Hay et al. (2011), ModelDB
139653, distinguendolo dal simulatore `hay_single_compartment` di questa repo.
Come riferimento principale viene usato `L5PCbiophys3.hoc`, cioe il modello di
Figura 4 vincolato sia sul BAC firing sia sul current-step firing. Le cinetiche
provengono direttamente dai file `.mod` originali.

Le costanti dei gate sono voltage-dependent. I numeri sotto sono stati calcolati
dalle equazioni originali a 34 °C, includendo il fattore
`q10 = 2.3^((34-21)/10)` quando presente. Gli intervalli sono ordini di grandezza
nel dominio -100..+50 mV, non una garanzia che ogni estremo venga visitato da
ogni compartimento.

## Inventario completo dei meccanismi intrinseci

| Meccanismo | Stati dinamici | Corrente/ruolo | tau caratteristica (ms) | Classe |
|---|---|---|---:|---|
| membrana/cavo | `v` | bilancio capacitivo, ionico e assiale | dipende da `cm`, conduttanze e geometria | veloce e accoppiata spazialmente |
| `NaTa_t` | `m`, `h` | Na transiente assonale/somatico | `m` 0.02–0.19; `h` 0.19–1.88 | ultrarapida |
| `NaTs2_t` | `m`, `h` | Na transiente somato-dendritico nel modello con inizio assonale | `m` 0.02–0.19; `h` 0.21–1.88 | ultrarapida |
| `Nap_Et2` | `m`, `h` | Na persistente | `m` 0.13–1.14; **`h` 427–2194** | `h` estremamente lenta |
| `K_Tst` | `m`, `h` | K transiente | `m` 0.12–0.43; `h` 2.7–19.3 | rapida/intermedia |
| `K_Pst` | `m`, `h` | K persistente | `m` 1.35–16.6; **`h` 122–414** | `h` molto lenta |
| `SKv3_1` | `m` | K rapido, ripolarizzazione | 0.92–3.60 | rapida |
| `Im` | `m` | M-current, adattamento | 0.02–51.3; massimo vicino a -35 mV | intermedia e voltage-dependent |
| `SK_E2` | `z` | K attivato dal Ca | 1 fisso | gate rapido, ma pilotato da `cai` lento |
| `Ca_LVAst` | `m`, `h` | Ca low-voltage activated | `m` 1.69–8.47; `h` 6.77–23.7 | intermedia |
| `Ca_HVA` | `m`, `h` | Ca high-voltage activated | `m` 0.24–7.04; **`h` 165–450** | `h` molto lenta |
| `CaDynamics_E2` | `cai` | accumulo/rimozione del Ca intracellulare | **460 soma; 122 apicale** in `L5PCbiophys3` | molto lenta |
| `Ih` | `m` | corrente HCN | 1.14–77.3; circa 42 a -70 mV | intermedia/lenta |
| `pas` | nessuno | leak | nessun gate | istantanea |

Valori rappresentativi a -70 mV:

| Stato | tau (ms) |
|---|---:|
| `h_Nap_Et2` | 2145 |
| `h_Ca_HVA` | 449 |
| `h_K_Pst` | 395 |
| `cai` somatico | 460 |
| `cai` apicale | 122 |
| `m_Ih` | 42 |
| `m_K_Pst` | 12.9 |
| `h_Ca_LVAst` | 22.8 |
| `m_Im` | 3.10 (puo salire a circa 51 vicino a -35 mV) |

La gerarchia lenta reale e quindi, in prima approssimazione:

```text
h_Nap  >>  cai_soma ~ h_CaHVA ~ h_KP  >  cai_apicale ~ Ih/Im  >  gate rapidi
```

`h_Nap` e la variabile intrinseca piu lenta del modello: intorno al riposo la
sua tau e circa 2 s. Non va confusa con `m_Nap`, che e rapida.

## Distribuzione regionale nel modello canonico `L5PCbiophys3`

Gli stati non esistono una sola volta per neurone: ogni segmento che contiene
un meccanismo possiede la propria copia dei suoi gate.

| Regione | Meccanismi dinamici |
|---|---|
| soma | `Ca_LVAst`, `Ca_HVA`, `SKv3_1`, `SK_E2`, `K_Tst`, `K_Pst`, `Nap_Et2`, `NaTa_t`, `CaDynamics_E2`, `Ih` |
| dendrite apicale | `Ih`, `SK_E2`, `Ca_LVAst`, `Ca_HVA`, `SKv3_1`, `NaTa_t`, `Im`, `CaDynamics_E2` |
| dendrite basale | `Ih` oltre a membrana passiva |
| assone troncato | membrana passiva in `L5PCbiophys3` |

Per un segmento somatico sono quindi presenti 15 gate, `cai` e `v`: 17 stati
intrinseci. Un segmento apicale ha 10 gate, `cai` e `v`: 12 stati. Un segmento
basale ha `m_Ih` e `v`. I voltaggi dei segmenti sono accoppiati dalle correnti
assiali; per questo il sistema completo non e una collezione di compartimenti
indipendenti.

`L5PCbiophys1`, `2`, `3` e `4` sono configurazioni diverse, non quattro copie
identiche del modello. Per esempio, il decadimento somatico del Ca vale 486,
376, 460 e circa 295 ms rispettivamente. `L5PCbiophys4` introduce inoltre
meccanismi attivi nell'assone e usa `NaTs2_t` nel soma/apicale. Quando si parla
di "modello Hay completo" occorre quindi dichiarare il file biophys usato.

## Sinapsi: originali e aggiunte

Le variabili AMPA/NMDA/GABA non fanno parte del nucleo intrinseco del rilascio
ModelDB 139653. Il file originale `epsp.mod` e un generatore di corrente con
`tau0 = 0.2 ms` e `tau1 = 3 ms`, senza stati di recettore da apprendere.

In una versione con sinapsi a doppia esponenziale, ogni recettore aggiunge due
stati (`A`, `B`). Le costanti usate nel nostro contesto esteso sono:

| Recettore | salita (ms) | decadimento (ms) |
|---|---:|---:|
| AMPA | 0.3 | 3 |
| NMDA | 2 | 70 |
| GABA-A | 0.2 | 8 |
| GABA-B, se attivo | 3.5 | 260.9 |

GABA-B sarebbe dunque un'altra variabile molto lenta, ma non e attivo nella
configurazione corrente e non appartiene al modello Hay intrinseco originale.

## Differenze rispetto al compartimento ridotto della repo

Il ridotto conserva 17 stati totali, ma non sono gli stessi 17 stati del soma
Hay originale.

| Aspetto | Hay somatico originale | Ridotto corrente |
|---|---|---|
| Na persistente | `m_Nap` e **`h_Nap`** | solo `m_Nap` |
| K transient/persistent | `m/h_KT` e `m/h_KP` | un solo `n_Kdr` |
| sinapsi | non intrinseche al modello base | `g_AMPA`, `g_NMDA`, `g_GABAA` |
| Ca decay | 460 ms nel modello canonico | 80 ms |
| `h_Ca_HVA` | circa 165–450 ms | circa 8–33 ms |
| `m_Ih` | circa 1–77 ms | circa 25–145 ms |
| `m_Im` | fino a circa 51 ms | circa 20–100 ms |

Il ridotto e quindi Hay-inspired, non una riduzione state-preserving. In
particolare elimina `h_Nap`, accorcia fortemente `cai` e `h_Ca_HVA`, ma rende
`Ih` e `Im` piu lenti in parte del dominio.

## Conseguenze per le finestre di rollout

Una finestra coerente con una variabile di tau `T` dovrebbe osservare almeno
3–5 tau per verificare il rilassamento e l'accumulo; per la fedelta di sistema
serve anche una finestra globale, perche gli errori dei gate rapidi possono
cambiare spike e Ca e contaminare le variabili lente.

| Obiettivo | Finestra indicativa per il Hay canonico |
|---|---:|
| spike e gate rapidi | 5–25 ms, con metriche event-aligned |
| gate intermedi / HCN / M | 100–300 ms |
| `cai`, `h_KP`, `h_CaHVA` | 1.5–2.5 s |
| `h_Nap` | almeno 6–10 s |
| stabilita statistica/regime | decine di secondi, senza richiedere identita punto-per-punto |

Di conseguenza, 500 ms resta una buona prima finestra globale per il ridotto
attuale, mentre per un futuro surrogato fedele al compartimento Hay originale
e solo una prova intermedia. Un test a 1 s e importante ma non esaurisce la
dinamica di `h_Nap`; per quella servono rollout multi-secondo.

## Fonti locali

- `139653/mod/*.mod`: equazioni e stati dei canali originali.
- `139653/models/L5PCbiophys3.hoc`: distribuzione regionale e parametri del
  modello canonico di Figura 4.
- `139653/models/L5PCtemplate.hoc`: morfologia, discretizzazione e accoppiamento
  dei compartimenti.
- `src/hay_single_compartment/simulator.py`: stato e cinetiche del ridotto.
- `src/hay_single_compartment/config.py`: costanti del ridotto e delle sinapsi.

