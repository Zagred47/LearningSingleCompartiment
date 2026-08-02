# Current research state — 2026-08-02

Current input-only models learn the common smooth manifold but collapse the rare fast voltage regime. In factorial 11 all four cells produce zero of 102 spikes and remain near or below -58 mV while the teacher reaches about +44.5 mV. High-frequency voltage power is nearly absent.

MR-STFT improves global, slow and subthreshold metrics, changes event RMSE by only about 0.19%, preserves zero recall and worsens mean synaptic-state NRMSE. The tested 7.5 ms causal frontend fails all important guardrails. These are scoped negative results, not blanket rejections of spectral loss or convolution.

Earlier diagnostics show broad event-support information reaches recurrent latents and gates. Residual experts sometimes localize events but either spill into subthreshold regions or fail to generate the amplitude. The unresolved fork is among exposure/imbalance, gradient conflict/optimization, exact phase accessibility and information-contract insufficiency.

DG-01 is complete. Event bins occupy 0.1275% of validation and explicit event-centred batches only 1.75% of optimizer updates, yet event-window gradients remain usable (median SNR 0.784; event-to-natural norm ratio 2.60). Target density versus conditional error has median rank correlation -0.944. This supports both exposure and imbalanced-regression explanations diagnostically, without separating them causally. Universal gradient conflict is weakened; block-specific recurrent conflict remains. Details are in [`loss_landscape_12_findings.md`](loss_landscape_12_findings.md).

Next: DG-02 on the exact DG-01 manifest. It tests whether broad event support, exact phase and voltage amplitude remain accessible in recurrent and decoder representations. Only then may the programme choose an objective/exposure factorial, predicted-state feedback or the capability bench.

The discovery program now also preserves the complete inductive-bias taxonomy and the creative methodology supplied by the user. After diagnosis, the capability phase will generate multiple small sketches, map capacity × difficulty surfaces and probe internal computation before any architecture is scaled. Evaluation includes granular state/regime/event/rollout metrics plus structured qualitative plot observations and statistics inside each network.
