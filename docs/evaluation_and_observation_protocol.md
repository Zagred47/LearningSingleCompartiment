# Evaluation and observation protocol

The previous metric set was too compact. Evaluation is now a tensor over state, family, compartment, regime, trajectory, event phase, temporal scale, rollout horizon, difficulty, capacity and seed. The canonical taxonomy is [`evaluation_taxonomy.json`](../research/evaluation_taxonomy.json).

## Four evaluation layers

1. **External behavior:** physical-state fidelity, residual distributions, event morphology, spectra, rollout, dynamical geometry, spatial propagation, calibration, robustness and efficiency.
2. **Internal computation:** activations, gates, representation geometry, probes, causal interventions and Jacobian/sensitivity analysis.
3. **Training/optimization:** learning curves, gradient/update flow, landscape geometry and stochastic stability.
4. **Capability profiles:** model capacity × task difficulty × data quantity, including compositional generalization.

No single score collapses these layers. An experiment has one primary decision metric but publishes the full relevant tensor and all guardrails.

## Granular behavioral battery

- Pointwise: RMSE/MAE/NRMSE, bias, correlation and worst coordinate.
- Residual morphology: quantiles, tails, skew/kurtosis, heteroscedasticity and autocorrelation.
- Events: precision/recall/F1, count, amplitude, timing, rise/decay, width, area, AHP and burst intervals.
- Frequency: PSD/coherence/cross-spectrum, STFT and wavelet-scale energy with phase.
- Rollout: error curves/AUC, first divergence, event survival, boundedness and family-specific drift.
- Dynamics: phase portraits, delay embeddings, recurrence quantification and local divergence.
- Spatial: source-target lag, attenuation, directionality and graph-distance error.
- Robustness: input rate/timing shifts, missing channels, state perturbations and OOD regimes.
- Efficiency: parameters, compute, memory, wall time and examples/updates to threshold.

## Statistics inside one network

For every relevant layer and regime, capture activation quantiles, entropy, sparsity, dead/saturated fractions and local derivative. For gated cells, record gate distributions and event-aligned trajectories. Track effective rank and singular spectra, latent similarity and recurrence, probe accessibility across lags, Jacobian spectra and gradient propagation.

Probes are followed by causal tests when they motivate an architectural claim. “The state is decodable” and “the prediction uses the state” are different statements.

## Mandatory qualitative plot review

Qualitative evidence is not decoration. Standard archives include:

- random validation trajectories selected before inference;
- preregistered worst cases;
- event-aligned waveform overlays and residual heatmaps;
- state-family × time and compartment panels;
- phase portraits and recurrence plots;
- horizon waterfalls;
- activation/gate event-aligned small multiples;
- all component-loss and guardrail learning curves.

Every observation is written before interpretation using: figure and selection rule, exact interval, literal morphology, affected states/regimes, possible mechanisms, metric blind spot, proposed measurable signature, alternatives and the next falsifying view.

Examples of legitimate observations are “predicted peaks remain clipped near −58 mV while timing support is approximately correct” or “the gate rises before teacher events but the residual remains uncorrelated.” “The model does not understand spikes” is an interpretation, not an observation.

Repeated morphology becomes a registered metric or diagnostic. A single attractive or catastrophic hand-picked trace never becomes a general claim.

## Loss landscape and partially independent views

Sharpness, anisotropy, connectivity, roughness/fractality and topology are not interchangeable. A surface may be smooth but contain disconnected minima, or sharp yet isotropic. Tier-1 estimators with clear actions come first; fractal, wavelet, minimum-energy-path and topological estimators enter only when they resolve a named remaining ambiguity.

The goal is not metric collection. Every indicator must state what it can distinguish, its estimator limitations and which intervention it may authorize.

