# Architecture research program

Architecture discovery moves from established broad biases to evidence-backed composition.

1. **Information contract:** input-only versus own predicted-state feedback; teacher state after initialization is forbidden.
2. **Temporal scaffold:** GRU/LSTM, TCN, S4, CfC/LTC, or explicit controlled vector-field models.
3. **Spatial scaffold:** monolithic four-compartment map versus shared local message passing.
4. **Regime structure:** switching/mixtures only after a fast expert demonstrates correction capability.
5. **Composition:** combine components only when each independently solves a required capability and the result beats a parameter-matched monolithic control.

The catalog and primary sources are in [`architecture_catalog.json`](../research/architecture_catalog.json).

Existing evidence rejects the sufficiency of the tested 7.5 ms CausalConv1d+GRU and the tested MR-STFT objective for spike recovery. It weakens an input-only gated residual mixture without a capable expert. It does not reject LSTM, S4, CfC, LTC, own-state feedback, message passing, or convolution under a different causal hypothesis.

Screens match parameters within a preregistered tolerance, optimizer updates, dataset, sampler, normalization, initialization distribution and evaluation manifest. Compute is reported. Three seeds support a claim; one seed is only preflight.

Activation, normalization, optimizer and regularization are conditional axes. They enter only after a trigger in [`indicator_catalog.json`](../research/indicator_catalog.json), preventing an endless Cartesian product.

