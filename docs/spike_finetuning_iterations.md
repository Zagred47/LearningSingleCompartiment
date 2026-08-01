# Spike fine-tuning experimental ledger

## Acceptance contract

A fine-tuned checkpoint is useful only if it improves rare spike dynamics while
remaining a faithful 61-state surrogate.  The validation selector therefore
rejects candidates that violate any of these constraints relative to GRU-MSE:

- normalized global MSE: at most `+5%`;
- subthreshold soma RMSE: at most `+10%`;
- samples above -20 mV: at most `2x` the teacher count;
- upward -20 mV crossings: at most `1.5x` the teacher count.

The baseline checkpoint is always retained as epoch 0.  If no trained epoch is
both admissible and better on the complete validation objective, the experiment
returns the baseline rather than presenting a degraded model as an improvement.

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
