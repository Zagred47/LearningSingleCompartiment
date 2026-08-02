# SR-01 predicted-state feedback findings

SR-01 is a one-seed, parameter-identical causal preflight. It tests whether
feeding a GRU its own previous 61-state prediction recovers fast phase that is
missing from an input-only latent. The result closes this direct-feedback
contract as an immediate remedy; it does not show that physical state or
state-space models are generally useless.

## Integrity

- Official archive SHA-256: `E131B33F42C11EC129A3BD7685A5EF40713A72ED2EE39D84698E302F20FCC691`.
- Dataset SHA-256: `1fd0eaf7ffc6bbd5e8eb2db64ba4bcc67289048ef0be9367760088ff1739a3bf`.
- All arms contain 330,461 parameters and have the same initial-state hash.
- Objective, sampler, optimizer-update schedule and initialization are matched.
- Teacher state enters only once at complete-trajectory initialization. It is
  never reinjected at a rollout or sampled-window boundary.
- Validation only; the test split remains unopened.

## Preregistered decision

| Arm | event soma RMSE (mV) | soma RMSE (mV) | subthreshold RMSE (mV) | mean state NRMSE | synaptic NRMSE | matched / 102 spikes |
|---|---:|---:|---:|---:|---:|---:|
| no state context | 17.3016 | 3.7640 | 1.1330 | 0.4198 | 0.1223 | 0 |
| initial state once | 17.2926 | 3.7326 | 1.0270 | 0.4057 | 0.1191 | 0 |
| own predicted-state feedback | 17.2815 | 3.7507 | 1.1053 | 0.4319 | 0.1787 | 0 |

Predicted feedback improves event RMSE by only `0.0642%` relative to the
initialization-only control, far below the preregistered 20% threshold, and
produces zero matched spikes. It violates the synaptic guardrail at `1.500x`;
mean-state, slow-state and subthreshold ratios are 1.065, 1.050 and 1.076.
Initial-state information alone improves event RMSE by only `0.0522%` versus no
state context. The preregistered branch is therefore
`close_direct_feedback_branch`.

## Granular and qualitative diagnostics

- Per-trajectory event-RMSE gains range from `-0.52%` to `+0.49%`. Six of eight
  trajectories have a positive sign, but the magnitude is negligible and two
  reverse it. This is not a hidden large subgroup effect.
- All predictions remain below approximately `-57.7 mV` at the soma while the
  teacher reaches `+44.5 mV`. Median predicted local maxima around teacher
  spikes remain approximately `-62 mV`.
- Event-specific RMSE remains 22.46 mV for isolated spikes, 21.52 for bursts,
  19.37 for rapid fire and 22.11 for spike-with-plateau under feedback. The
  visible failure is complete amplitude suppression, not merely jitter.
- Feedback and initialization-only soma traces remain highly correlated
  (`r=0.949`). Feedback changes the trajectory but not the missing phenotype.
- Feedback helps a few endogenous gates, most strongly `soma.h_Nap_Et2`
  (normalized error ratio 0.555), while strongly degrading exogenous synaptic
  traces: several AMPA, NMDA and GABA coordinates worsen by 1.5--1.9x.

The last point explains why unrestricted feedback is a poor information
contract: it recursively mixes self-generated error into state coordinates
whose causal evolution is already determined by external event inputs. This is
evidence for later structured state decomposition, but it does not authorize
inventing a masked or reduced GRU without an independent capability test.

## Scope and protocol note

The archive stores aggregate best-checkpoint metrics and float16 validation
traces. The preregistration named peak-amplitude error as a secondary metric,
but the run did not write it into the official summary table. It can be
reconstructed diagnostically from the traces (approximately 77 mV local peak
MAE for every arm), but that reconstructed value is not used in the formal
decision. The result is one seed and closes only the tested direct all-state
feedback implementation.

## Consequence

Do not replicate SR-01 and do not scale this feedback cell. Enter the
capability bench before training another large Hay surrogate. CB-01 tests ten
published architecture families on an atomic synthetic process that combines
slow evidence, a state-dependent trigger, a narrow fast transition and
recovery. Only diverse hits that survive capacity x difficulty scaling are
eligible for mechanism deconvolution and a later Hay preflight.
