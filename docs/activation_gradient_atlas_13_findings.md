# DG-02 — Activation and Gradient Atlas findings

DG-02 selects the preregistered `information_contract_or_scaffold` branch. The frozen input-only models reliably encode broad event support, but exact fast phase and event-voltage amplitude are not trajectory-generalizable through the tested linear probes. This is a representation/readout diagnosis, not proof that the information is absent under every nonlinear probe.

## Integrity and contract

- Archive SHA-256: `e06370b5437321ab4c67463d2f82b79b315e53a2737f5c44db8ea7bb3ec3c58e`.
- Dataset SHA-256: `1fd0eaf7ffc6bbd5e8eb2db64ba4bcc67289048ef0be9367760088ff1739a3bf`.
- DG-01 manifest SHA-256: `19da693936a75ab6cfb00358416db1e130477400afdea1007b40b156e4783971`.
- Four frozen factorial-11 checkpoints; zero optimizer steps; validation only; test unopened.
- Six event windows come from three trajectories and two pairs overlap. Duplicate trajectory/time coordinates were removed before probing.
- Ridge penalty `alpha=100` was fixed before execution; feature scaling used only training trajectories inside each leave-one-trajectory-out fold.
- Manual GRU gate replay agrees with the model to maximum absolute error between `1.6e-6` and `9.8e-6`.

## Preregistered branch decision

The aggregate decision statistics are:

| statistic | observed | relevant threshold |
|---|---:|---:|
| median hidden event-support AUROC | 0.879 | at least 0.7 |
| median hidden exact-phase correlation | 0.111 | below 0.2 |
| median decoder exact-phase correlation | 0.143 | below 0.2 |
| median hidden event-residual improvement | -45.5% | below +5% |
| median decoder event-residual improvement | -74.5% | below +5% |

This matches the preregistered information-contract/scaffold signature. The objective/exposure branch is not selected as the immediate next intervention, although DG-01's imbalance and exposure evidence remains valid and deferred rather than falsified.

The archive reports out-of-fold predictions only through aggregate metrics and does not preserve one row per held-out trajectory. Consequently the preregistered fold-instability clause cannot be audited directly from this artifact. The branch is strong enough to authorize one reversible L1 causal preflight, but not a confirmatory representation claim.

## What is represented

Broad event support is strongly decodable from the hidden state: AUROC is 0.894 and 0.909 for the two GRUs, and 0.864 and 0.769 for the two causal-ConvGRUs. The decoders retain part of this signal (0.701–0.887). Raw packed spikes alone have AUROC 0.370 under trajectory holdout.

The causal convolutional frontend itself creates broad-support information: its recurrent-input AUROC is 0.811–0.849, versus 0.361–0.369 for the GRU input encoder. Yet the ConvGRU recurrent/readout path can erase part of it, especially under MR-STFT (0.849 frontend → 0.769 hidden → 0.701 decoder). Thus the tested convolution is not simply useless: it computes a diagnostic feature that the complete scaffold does not convert into a correct waveform.

## What is not accessibly represented

Exact phase correlations are weak. Hidden correlations range from 0.028 to 0.205; decoder correlations range from 0.041 to 0.252. The causal models shift some coarse phase information toward the decoder, but remain far below the preregistered 0.5 readout-branch threshold.

No representation yields a trajectory-held-out linear correction of the event-voltage residual. Hidden probes worsen RMSE by 6.3–53.8%; decoder probes worsen it by 7.4–92.6%. This rules out the simple explanation that a correct spike-amplitude correction is already linearly present and merely ignored by the final affine map.

## Internal dynamics

- Hidden participation ratios are only 2.55–3.55, with 90% variance ranks of 4–7. Decoder participation ratios are 2.09–3.26. This is severe empirical compression relative to the nominal hidden width, but it is not by itself proof of insufficient theoretical capacity.
- Update gates are strongly polarized. The causal models place roughly 62–73% of update-gate values below 0.05 or above 0.95, while the plain GRUs place roughly 40–46% above 0.95.
- The causal-ConvGRU+MR-STFT cell shows the clearest regime-dependent retention proxy: about 25.7 steps in event windows, 67.9 in natural windows and 79.3 in subthreshold windows. Adaptive timescale behavior therefore exists, but does not produce correct spikes.
- Event-window activation-gradient norms exceed subthreshold norms by 18–108×, depending on model and layer. Near-zero gradient fractions are negligible. Classical dead activations or globally vanished event gradients are not the dominant signature.

## Scoped scientific conclusion

For the current 61-state, four-compartment, input-only contract and these single-seed checkpoints, the networks learn a low-dimensional smooth/event-support manifold but not a trajectory-generalizable coordinate for exact fast phase and amplitude. The failure is therefore upstream of a simple readout replacement and is not explained by globally dead gradients. This authorizes a controlled information-contract intervention, not arbitrary capacity scaling or another unmotivated loss.

The next causal test is SR-01: compare parameter-identical standard GRU scaffolds with no physical-state context, initialization-only physical state, and recursively predicted-state feedback. Teacher state may enter only once at trajectory initialization; the feedback arm must consume only its own normalized prediction thereafter. A one-seed preflight is allowed before any three-seed replication.

## Limitations

- Linear probes test accessible linear geometry, not all nonlinear decodability and not causal use.
- There is one trained seed per factorial cell.
- Event evidence is concentrated in three trajectories.
- Per-held-out-trajectory probe scores were not exported, so fold-sign stability cannot be independently checked.
- Effective-rank estimates combine temporally correlated windows and should be treated as a comparative indicator.
- Gate time constants are proxies derived from update values, not identified physical time constants.
