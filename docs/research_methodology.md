# Protocollo di scoperta architetturale per il micro-Hay

## Obiettivo

Non cerchiamo una successione di modifiche che abbassi casualmente una metrica.
Cerchiamo esperimenti che distinguano ipotesi sulla struttura astratta della
dinamica causale: memoria, multiscala temporale, localita, composizione,
separazione dei regimi e geometria dello stato. Ogni risultato deve restringere
lo spazio delle spiegazioni anche quando il modello candidato perde.

Il teacher e noto durante la progettazione, ma l'interpretazione dei risultati
segue una finta ignoranza controllata: un'architettura viene introdotta solo se
esiste gia nella letteratura o come composizione standard, e ogni conclusione
si limita alla proprieta effettivamente isolata dall'ablation.

## Unita sperimentale

Una riga del registro corrisponde a una domanda falsificabile. Prima del run si
bloccano:

1. baseline e candidato;
2. unica differenza causale principale;
3. metrica primaria e guardrail;
4. budget di parametri, esempi, update e seed;
5. criterio di promozione, reiezione e risultato nullo;
6. fonte scientifica che giustifica il candidato.

Il dataset canonico resta immutato. Maschere di eventi, finestre, embedding di
Takens e recurrence plot sono viste diagnostiche: non autorizzano a cambiare il
test o a campionarlo in modo favorevole.

## Separazione discovery/confirmation

- `train`: stima dei parametri.
- `validation`: diagnosi, scelta dell'ipotesi successiva e selezione del
  checkpoint.
- `test`: conferma una tantum dopo aver bloccato candidato e regola decisionale.

Se il test viene consultato durante l'iterazione, diventa di fatto validation e
serve un nuovo holdout indipendente per fare affermazioni confermative.

## Assi da esplorare

| Asse | Domanda astratta | Controllo minimo |
|---|---|---|
| Scaffold/layer | Quale famiglia di operatore rappresenta meglio la memoria causale? | budget e training pareggiati |
| Architettura | La dinamica e monolitica o beneficia di sottosistemi interagenti? | controllo di capacita monolitico |
| Attivazione | Saturazione, smoothness o supporto non limitato ostacolano il flusso? | stessa rete, sola attivazione diversa |
| Loss | Quale errore osservabile allinea davvero l'ottimizzazione al rollout? | stessa architettura e sampler |
| Regolarizzazione | Il fallimento deriva da capacita, co-adattamento o instabilita? | stessa loss e stesso numero di update |
| Normalizzazione | Le scale fisiche alterano il condizionamento o cancellano stati lenti? | invertibilita e fit solo sul train |
| Training/optimizer | Il candidato non funziona o non viene raggiunto dall'ottimizzazione? | sweep piccolo e preregistrato |
| Stocasticita | Quanto dipende il risultato da seed, ordine e rumore? | almeno tre seed e intervalli |

Non si esegue un factorial sweep indiscriminato. Il Failure Atlas identifica la
classe dominante del residuo; quella diagnosi sceglie il prossimo asse.

## Pipeline a porte

### Porta 0 - Integrita

Verifica hash del dataset, schema, assenza di teacher forcing, split per
traiettoria, causalita, equivalenza chunk/full e corretto ripristino del
checkpoint. Un fallimento qui invalida tutto il run.

### Porta 1 - Failure Atlas

Il baseline viene descritto senza training nuovo mediante errori per stato,
famiglia, compartimento, evento e orizzonte; spike timing/forma; spazio delle
fasi; recurrence; spettro e autocorrelazione del residuo; attivazioni, gate e
rango effettivo del hidden.

L'Atlas produce pattern osservativi, non diagnosi causali. Esempio: update gate
saturi insieme a errori rapid-fire supportano un test sulla memoria rapida, ma
non dimostrano che il gate sia la causa.

### Porta 2 - Scaffold screen

Con loss MSE e protocollo invariati si confrontano poche famiglie gia
giustificate: GRU, ConvGRU, ConvLSTM, CfC/LTC se l'implementazione ufficiale e
disponibile. Lo scopo e scegliere uno scaffold, non ottimizzarlo.

Nota di nomenclatura: le classi storiche `InputOnlyConvGRU` e
`InputOnlyConvLSTM` della repo non implementano la convoluzione dentro le
transizioni ricorrenti delle architetture spaziali ConvGRU/ConvLSTM originali.
Sono composizioni causali `Conv1d dilatata -> GRU/LSTM standard`. Nei report
nuovi vengono chiamate `CausalConv1d+GRU` e `CausalConv1d+LSTM`; rinominare il
concetto evita di attribuire al test un bias che il codice non possiede.

### Porta 3 - Diagnosi dell'ottimizzazione

Sul solo scaffold scelto si misurano gradienti per blocco, curvature proxy,
sensibilita a perturbazioni normalizzate e interpolazioni 1D/2D della loss.
Questi test separano un bias inadatto da un problema di ottimizzazione.

### Porta 4 - Ablation ortogonali

Si cambia un asse alla volta nell'ordine suggerito dall'Atlas. Ogni candidato
ha un controllo di capacita e una previsione esplicita su quali metriche devono
migliorare e quali devono restare invariate.

### Porta 5 - Conferma e scaling

Solo le varianti promosse passano a tre o piu seed, curve di data efficiency,
rollout lunghi e test holdout. Lo scaling viene dopo la conferma del bias, non
prima.

## Loss landscape

Una visualizzazione della loss non e una mappa assoluta della funzione: dipende
dalla parametrizzazione e presenta simmetrie. Usiamo direzioni normalizzate per
filtro/blocco e riportiamo almeno:

- interpolazione dal checkpoint iniziale a quello finale;
- perturbazioni casuali normalizzate a piu raggi;
- sharpness relativa entro un raggio dichiarato;
- trace/Frobenius proxy dell'Hessiano con seed fissato;
- gradient norm e update-to-weight ratio per blocco.

La stessa minibatch congelata serve per confrontare varianti; il risultato va
letto insieme a validation e rollout, mai come criterio unico.

## Grounding scientifico

Ogni idea entra nel registro bibliografico con fonte primaria, meccanismo
proposto, previsione osservabile e differenza rispetto al nostro setting. Una
fonte giustifica un esperimento, non il suo esito. Le ricerche web devono
precedere l'implementazione del candidato e privilegiare paper e documentazione
ufficiale.

## Pacchetto minimo di un run

Ogni output deve contenere: contratto e hash, card pre-run, configurazione
risolta, commit Git, ambiente, seed, history completa, best/last checkpoint,
metriche machine-readable, grafici diagnostici e card post-run con decisione.
Un archivio senza queste informazioni e esplorativo, non confrontabile.
