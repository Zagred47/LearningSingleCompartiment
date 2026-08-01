# FA-00 - GRU-MSE Failure Atlas findings

## Integrity

The read-only replay used validation only: 8 trajectories, 10,000 model steps,
61 states and 0.5 ms resolution. Dataset and checkpoint SHA-256 matched. The
test split was not consumed.

## Dominant observation

The teacher produced 102 somatic spikes and the GRU produced zero. Somatic
RMSE was 0.965 mV in subthreshold samples but 19-22 mV in isolated, burst,
rapid-fire and spike-plus-plateau samples. Synaptic states remained accurate
(mean normalized RMSE 0.111; mean correlation 0.995).

The power ratio prediction/teacher was 1.020 below 10 Hz, 0.499 at 10-50 Hz,
0.0123 at 50-200 Hz and 0.00184 at 200-1000 Hz. The largest state errors were
fast somatic gates (`m_NaTa_t`, `m_Ca_HVA`, `m_K_Tst`, `m_SKv3_1`) rather than
the slow synaptic traces.

## Internal representation

Reset and update gates were not globally dead. Update-gate high saturation
decreased from about 40% in subthreshold samples to about 34% in fast events,
and hidden/decoder activation distributions changed during burst, rapid-fire
and plateau windows. This supports, but does not prove, that event information
reaches the recurrent representation before the final continuous-state output.

The hidden effective rank was 12 at 90% and 44 at 99% out of 200 dimensions.
Because validation is 95% subthreshold and the teacher itself may have a
low-dimensional attractor, this is not sufficient evidence for a capacity
bottleneck.

## Falsifiable next step

The next preflight is a preregistered 2x2:

| | MSE | MSE + MR-STFT |
|---|---|---|
| GRU | baseline | objective main effect |
| CausalConv1d+GRU | architecture main effect | interaction |

All cells train from scratch with the same data order, windows, updates and
validation selection metric. A cell advances only with at least 15% lower
event soma RMSE and no more than 10% degradation in global/slow normalized
RMSE or 25% degradation in subthreshold soma RMSE. This is a one-seed
preflight; any promoted contrast requires at least three seeds before a claim.

