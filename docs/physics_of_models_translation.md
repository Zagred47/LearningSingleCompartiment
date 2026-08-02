# Translating Physics of Language Models to the Hay surrogate

The supplied talks are about language models; the methodology, not their domain-specific mechanisms, is transferred here.

| Physics-of-models object | Hay translation |
|---|---|
| intelligence decomposed into atomic skills | slow integration, conditional spike transition, refractory recovery, long adaptation, spatial propagation, stable rollout |
| synthetic pretraining playground | controlled dynamical microtasks with known generators and adjustable difficulty |
| model size × reasoning difficulty mini-scaling law | state/parameter budget × delay, event rarity, threshold depth, coupling distance or rollout horizon |
| behavioral process | predicted physical states, events, spectra, phase portraits and rollout |
| mental process | activations, gates, latent geometry, probes, gradients, Jacobians and causal interventions |
| benchmark contamination | trajectory leakage, teacher-state leakage, event-window overlap and reuse of test information |
| knowledge storage/extraction/manipulation | memory retention, accessibility/readout and state-dependent transformation |
| backward feature correction | whether deeper blocks refine rather than erase useful state/event features |
| Canon horizontal flow | general question of where local causal information must be directly accessible—not an instruction to add a convolution |
| synthetic-to-real resonance | capability-bench mechanism survives matched training on the 61-state teacher and later across scale/data |

## Mini-scaling surfaces for this project

Every capability family should vary at least two axes:

- model state or parameter budget;
- task difficulty: delay, number of interacting timescales, threshold sharpness, event rarity, graph distance, noise or horizon;
- optionally data quantity/event count as a third axis.

We seek transition boundaries: where a mechanism begins to learn reliably, how the boundary shifts with data, and whether seed variance is smaller than the architectural effect. A single large model succeeding on an easy task provides little architectural information.

## Learnability tests

An architecture can express a capability yet fail to learn it. Distinguish:

1. **expressive upper bound:** oracle features, teacher-state or overfit experiment, explicitly labeled as privileged;
2. **optimization:** same architecture under controlled objectives/samplers/optimizers;
3. **representation presence:** held-out probes with capacity controls;
4. **causal use:** ablation or intervention changes the relevant output;
5. **generalization:** the learned mechanism crosses difficulty, data and rollout regimes.

This ladder prevents “the network could represent it” and “a probe decoded it” from becoming premature explanations.

## Experimental controls inherited from the talks

Architecture comparisons must align dataset, order, paired seeds, code path, information contract, depth/width when comparable, parameter/update budgets and hyperparameter quality. Nearby learning-rate robustness complements multi-seed replication. Final tables are accompanied by learning curves and capability surfaces because delayed emergence and seed-dependent ranking are themselves scientific results.

