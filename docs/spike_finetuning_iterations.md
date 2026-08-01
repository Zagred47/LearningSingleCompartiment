# Spike fine-tuning experimental ledger

## Acceptance contract

A fine-tuned checkpoint is useful only if it improves rare spike dynamics while
remaining a faithful 61-state surrogate.  The validation selector therefore
rejects candidates that violate any of these constraints relative to GRU-MSE:

- normalized global MSE: at most `+5%`;
- subthreshold soma RMSE: at most `+10%`;
- samples above -20 mV: at most `2x` the teacher count;
- upward -20 mV crossings: at most `1.5x` the teacher count.
- mean normalized RMSE across the 61 individual states: at most `+5%`;
- mean normalized RMSE of slow calcium, Ih, Im, SK and NMDA-decay states:
  at most `+5%`.

The baseline checkpoint is always retained as epoch 0.  If no trained epoch is
both admissible and better on the complete validation objective, the experiment
returns the baseline rather than presenting a degraded model as an improvement.
The final archive also contains `statewise_rmse.csv`, so individual-state
degradation remains visible even when group means pass.

## Iteration 03: conservative rare-event objective (rejected)

Observed held-out result:

- soma RMSE: `3.566 -> 11.272 mV`;
- mean normalized 61-state RMSE: `0.388 -> 0.425`;
- spike precision/recall/F1: `0.263 / 0.147 / 0.189`;
- time above -20 mV: `0.089 s` teacher versus `1.403 s` prediction;
- only 13 of 61 state-wise RMSE values improved.

The positive-class-weighted threshold BCE and one-sided peak deficit admitted a
shortcut: a broad depolarized plateau received much of the reward of a spike
without matching its narrow waveform.  The weighted validation sum also allowed
global and subthreshold quality to deteriorate.

## Iteration 04: waveform-constrained objective (pending Kaggle result)

Changes are deliberately limited to the objective and checkpoint selector; the
GRU architecture, pure spike inputs, dataset, hidden-state feedback and complete
61-state supervision remain unchanged.

The new objective uses established regression constraints:

- symmetric full-state MSE in teacher spike neighbourhoods;
- symmetric physical soma-voltage waveform MSE;
- first-derivative matching (a first-order Sobolev loss);
- symmetric soft threshold-occupancy matching with no class weighting;
- functional distillation from the frozen GRU-MSE outside spike windows.

It removes the asymmetric peak deficit, positive-weighted BCE and separately
overweighted rapid-gate term.  Stratified event windows remain the mechanism for
showing rare trajectories more often; the loss no longer changes their target
class prior.

## Real-data preflight evidence

A diagnostic subset was formed from the real HDF5 using 6 event-rich training,
2 validation and 2 test trajectories.  This is not a replacement benchmark; it
was used only to falsify mechanisms before spending a full Kaggle run.

- v2, all parameters trainable: every epoch violated the global constraint and
  no predicted crossing appeared;
- v3, only the existing decoder trainable: global and subthreshold behaviour
  were preserved, but spike waveform error changed by less than 0.1% and no
  crossing appeared;
- v4, all parameters plus teacher-derived core/slope weighting: weighted
  waveform error fell by about 18%, but the subthreshold soma RMSE increased
  from 2.29 mV to more than 4.3 mV and no crossing appeared.

The v4 full-state reference term stayed numerically small while soma voltage
degraded because its error was averaged with 60 other states.  This motivates a
separate physical soma-reference constraint outside teacher event windows.

## Latent-state probes

A frozen-GRU linear probe was trained only for diagnosis.  On held-out real
trajectories it obtained:

- suprathreshold occupancy: ROC-AUC 0.975, average precision 0.0525 versus a
  0.0024 prevalence;
- a +/-2 ms crossing neighbourhood: ROC-AUC 0.979, average precision 0.318
  versus a 0.0153 prevalence.

Pointwise MLP and causal linear probes using 1--8 ms of hidden-state history did
not improve exact occupancy average precision.  The latent therefore identifies
the slow spike regime well but does not linearly encode the exact rapid phase.

## Iteration 05: auxiliary phase supervision (rejected)

The GRU, input contract and 61-state decoder remain unchanged at inference.  A
training-only standard linear auxiliary head is attached to the recurrent
sequence and jointly predicts suprathreshold occupancy and local voltage
derivative.  This is deep supervision: it forces rapid phase information into
the latent without converting classifier probability into membrane voltage.

The physical objective retains symmetric waveform/derivative matching and adds
an explicit soma-voltage distillation term outside spike windows.  The auxiliary
head is saved for audit but is not required for inference.  Checkpoint admission
still uses only physical rollout metrics and the original hard constraints.

Held-out evidence shows that the auxiliary task itself was learned, but did not
produce the required physical waveform:

- auxiliary occupancy BCE fell from about `0.726` to `0.158`;
- the selected checkpoint remained epoch 0 (the unchanged GRU-MSE);
- the last candidate produced only 5 predicted crossings for 136 teacher spikes,
  with precision `0.20`, recall `0.007` and F1 `0.014`;
- soma RMSE rose from `3.566` to `6.286 mV`, while subthreshold RMSE rose from
  `1.076` to `2.528 mV`;
- the prediction formed a broad depolarized hump instead of narrow rapid phases.

This falsifies the hypothesis that stronger pointwise/deep supervision alone is
enough.  The frozen GRU latent identifies a spike regime, but its shared
recurrent/pointwise decoding path does not express the required fast phase.

## Iteration 06: frozen GRU plus causal residual TCN (Kaggle candidate)

The next experiment changes one architectural assumption while preserving the
input contract and the complete 61-state target:

- the converged GRU-MSE is fully frozen;
- a standard causal dilated Conv1d stack receives the frozen GRU sequence plus
  the same packed spike inputs;
- dilations `(1, 2, 4, 8)` and kernel size 3 give a 31-step (`15.5 ms`)
  receptive field;
- a zero-initialized 1x1 projection predicts a residual for every one of the 61
  states, so epoch 0 is exactly GRU-MSE;
- only the residual adapter is optimized; no teacher or physical predicted state
  is fed back, and the auxiliary classifier is removed to isolate the temporal
  architecture hypothesis.

The same physical waveform objective, checkpoint constraints, selected-versus-
last audit and individual/slow-state audit remain in force.  A positive result
would support the abstract claim that the system contains a fast, finite-memory
correction superimposed on slower recurrent dynamics.  A rejected result would
falsify that specific decomposition without damaging the baseline checkpoint.
