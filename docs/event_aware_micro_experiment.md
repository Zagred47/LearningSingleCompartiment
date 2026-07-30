# Micro-Hay event-aware experiment

## Invarianti

- Il teacher resta il micro-neurone Hay a quattro compartimenti.
- Gli ingressi del surrogate sono esclusivamente spike presinaptici binari.
- Nessuno stato fisico del teacher entra nel modello, neppure durante il burn-in.
- Tutti i 61 stati sono target supervisionati.
- Il test naturale non viene riequilibrato.

## Catalogo della risposta

`classify_micro_events` produce etichette multi-label per subthreshold,
quasi-soglia, spike isolati, burst, rapid-fire, plateau di tuft, spike durante
plateau e recupero post-spike. Le classi sono ricavate dalla risposta simulata,
non dal nome del regime di input.

Il dataset HDF5 conserva traiettorie intere e lunghe. Questo evita di perdere
calcio, NMDA, Im, Ih, adattamento e recupero. L'arricchimento avviene in due
modi controllati:

1. un pool train più ampio aumenta la varietà causale senza alzare globalmente
   i rate sinaptici;
2. ogni epoca contiene sia il passaggio naturale completo sia finestre
   event-centered estratte secondo una miscela dichiarata.

Ogni finestra stratificata riceve 2000 ms di contesto precedente, costituito
solo da spike. La rete costruisce quindi autonomamente lo stato nascosto prima
della parte supervisionata.

## Obiettivo

La loss event-aware somma:

- MSE globale sui 61 stati normalizzati;
- errore del soma nelle finestre attorno agli spike;
- errore sulla derivata del soma;
- errore dei gate somatici rapidi nelle stesse finestre;
- classificazione soft del superamento della soglia somatica.

La GRU-MSE di controllo usa lo stesso dataset e lo stesso sampler. In questo
modo il confronto GRU-MSE/GRU-event isola la loss, mentre il confronto fra
architetture usa lo stesso obiettivo event-aware.

## Modelli

- GRU standard: 318261 parametri.
- Branch ELM: dinamica ELM preesistente, scalata a circa 318k parametri e con
  testa a 61 stati.
- ConvGRU: Conv1D temporali causali dilatate seguite da una GRU standard.
- ConvLSTM: lo stesso front-end seguito da una LSTM standard.

Le convoluzioni conservano una cache degli input pari al receptive field;
spezzare una traiettoria in chunk non resetta il contesto causale.

## Valutazione

`comparison.csv` contiene metriche naturali complessive, spike count,
precision/recall esatte e RMSE somatico per ogni classe. I checkpoint includono
pesi, normalizzazione, ordine degli stati/input, configurazione, hash del
dataset e catalogo degli eventi. Gli archivi delle predizioni permettono una
valutazione successiva con tolleranze temporali diverse senza riaddestrare.
