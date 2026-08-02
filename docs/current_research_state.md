# Current research state — 2026-08-02

Current input-only models learn the common smooth manifold but collapse the rare fast voltage regime. In factorial 11 all four cells produce zero of 102 spikes and remain near or below -58 mV while the teacher reaches about +44.5 mV. High-frequency voltage power is nearly absent.

MR-STFT improves global, slow and subthreshold metrics, changes event RMSE by only about 0.19%, preserves zero recall and worsens mean synaptic-state NRMSE. The tested 7.5 ms causal frontend fails all important guardrails. These are scoped negative results, not blanket rejections of spectral loss or convolution.

Earlier diagnostics show broad event-support information reaches recurrent latents and gates. Residual experts sometimes localize events but either spill into subthreshold regions or fail to generate the amplitude. The unresolved fork is among exposure/imbalance, gradient conflict/optimization, exact phase accessibility and information-contract insufficiency.

Next: DG-01 then DG-02 on frozen checkpoints and identical validation batches. Their signatures—not preference—select matched exposure, Balanced MSE, conflict treatment, predicted-state feedback or the capability bench. See [`research_operating_system.md`](research_operating_system.md).

