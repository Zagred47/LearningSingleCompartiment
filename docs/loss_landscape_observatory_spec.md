# DG-01 — Loss Landscape Observatory

DG-01 is next. It performs no training and never opens the test split. It distinguishes target imbalance, weak event-gradient exposure, cross-regime gradient conflict and a smooth low-frequency optimization basin.

## Frozen inputs

- four factorial-11 checkpoints: GRU-MSE, GRU-MRSTFT, CausalConvGRU-MSE and CausalConvGRU-MRSTFT;
- dataset SHA-256 `1fd0eaf7ffc6bbd5e8eb2db64ba4bcc67289048ef0be9367760088ff1739a3bf`;
- one saved validation manifest with equal natural, event-centred, subthreshold, slow-state and synaptic-state views;
- identical normalization and deterministic evaluation mode; no test trajectories.

## Tier-1 measurements

1. Per-regime loss, gradient norm, variance and signal-to-noise ratio.
2. Pairwise gradient cosine by regime and parameter block, with a loss-scale-normalized companion view.
3. Event-gradient contribution under actual sampling and equal-count diagnostic batching.
4. Filter-normalized initial-to-final and paired-checkpoint interpolation.
5. Filter-normalized local 1-D/2-D slices at fixed radii and multiple direction seeds.
6. Hessian top eigenvalue with convergence residual.
7. Hutchinson Hessian trace with probe-count convergence and confidence interval.
8. Perturbation sensitivity of amplitude, spectrum, common-regime and slow-state losses.

Every surface is regime-conditioned. A single aggregate surface cannot answer the suspected conflict.

| Signature | Interpretation | Authorized branch |
|---|---|---|
| actual contribution tiny; equal-count SNR acceptable | exposure deficit | TR-01 |
| stable negative event versus slow/synaptic cosine | objective conflict | LO-04 |
| event gradient strong/aligned but final model smooth | basin/optimizer path | OP-01 |
| target density predicts error with usable gradients | imbalanced regression | LO-02 |
| no useful event gradient in shared representation | representation/information deficit | DG-02 then SR-01 |
| signature unstable over batches/directions | inconclusive | improve estimator, no new training |

Outputs: manifests and hashes, raw per-example/regime summaries, cosine matrices, interpolation/slice data, Hessian convergence logs, figures and `decision.json` naming the updated knowledge/hypothesis IDs.

Mode connectivity, minimum-energy paths, wavelet/fractal estimators and persistent homology are Tier 2. They are allowed only when Tier 1 leaves a named ambiguity and cannot authorize architecture changes alone.

