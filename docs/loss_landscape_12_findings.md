# DG-01 — Loss Landscape Observatory findings

## Decision

DG-01 supports two coupled diagnostic explanations: rare-event exposure is
too small, and the regression objective is dominated by dense subthreshold
targets. It does **not** causally distinguish these factors. Universal gradient
conflict is weakened, while architecture- and parameter-block-specific conflict
remains possible. DG-02 must inspect whether exact event phase and amplitude are
present and usable inside the frozen networks before any training intervention.

Artifact SHA-256:
`195BC64B360114527398CC06562E78DB4461DDE3489972D5568664BD55EE4540`.
Dataset SHA-256:
`1fd0eaf7ffc6bbd5e8eb2db64ba4bcc67289048ef0be9367760088ff1739a3bf`.
No optimizer step was taken and the test split remained closed.

## What the experiment measured

- four frozen factorial-11 checkpoints;
- one deterministic 18-window validation manifest;
- six natural, six event-centred and six strictly subthreshold windows;
- loss, gradient norm and gradient SNR for five objective views;
- gradient cosine globally and by input/frontend, recurrent and decoder block;
- filter-normalized local event-loss surfaces;
- initial-to-final and paired-objective interpolation;
- Hessian dominant curvature and trace estimates on two windows per view;
- target-voltage density and conditional prediction error.

## Strong signatures

Event bins occupy only `0.1275%` of validation. Explicit event-centred batches
account for `8/456 = 1.75%` of factorial-11 optimizer updates. When event
windows are presented directly, however, their gradients are not absent:
median SNR is `0.784`, and the median event-to-natural mean-gradient norm ratio
is `2.60`. The four ratios are `1.83`, `2.44`, `2.76` and `14.04`.

Target density and conditional voltage RMSE have median rank correlation
`-0.944`. For GRU-MSE, the dense `[-70,-66)` mV bin contains 32,555 samples and
has RMSE `0.92` mV. The `[20,60)` mV bin contains 39 samples and has RMSE
`97.89` mV. All four models show the same morphology.

Initial-to-final paths reduce subthreshold loss by roughly `87–95%` and reduce
synaptic loss strongly, but event loss falls by only about `9–15%`. Moving from
MSE to MR-STFT also leaves the event path nearly flat. This matches the earlier
failure atlas: training allocates useful capacity to dense smooth regimes while
rare voltage excursions remain almost unchanged.

## Gradient conflict

The global median event-versus-slow cosine is `-0.069`; event-versus-synaptic
is `-0.036`. These miss the preregistered universal-conflict threshold of
`-0.10`. A universal PCGrad-style intervention is therefore not authorized.

Conflict is nevertheless structured. In the two GRU checkpoints, recurrent
event-versus-slow cosines are `-0.357` and `-0.465`; GRU-MRSTFT also has an
input/encoder cosine of `-0.694`. Decoder gradients are mostly orthogonal, and
the causal frontend changes the pattern. DG-02 must therefore preserve block-
and-layer-specific measurements rather than collapse everything into one
cosine.

## Landscape and curvature cautions

The local event surfaces are nearly planar over a filter-normalized radius of
`0.02`; their total range is only about `0.09–0.35%` of centre loss. They do not
show a useful sharp local event minimum. Random two-dimensional slices cannot
prove global connectivity or expressivity.

Most power iterations converge, but the GRU-MSE event estimate does not:
relative residual is `1.87`. Hutchinson trace uncertainty is also large for
several event views (`21–60%` relative standard error from four probes). These
curvature values are descriptive and cannot select the next branch.

## Limitations

- All checkpoints originate from one training seed.
- Six event windows come from only three trajectories; two pairs overlap.
- Gradient availability does not prove that the hidden state contains exact
  causal phase or sufficient waveform information.
- Density, sampler exposure and scalar objective weighting remain correlated.
- Interpolation uses a recomputed 512-step truncated context rather than the
  full optimization trajectory.

## Next falsification

Run DG-02 on the exact manifest SHA-256
`19da693936a75ab6cfb00358416db1e130477400afdea1007b40b156e4783971`.
Measure hidden and decoder activation distributions, GRU gates, effective rank,
phase/amplitude probes and causal interventions. If exact phase/amplitude is
available and used, the efficient causal follow-up is a preregistered `2 x 2`
factorial separating event exposure from balanced regression. If information is
present but not used, test the readout contract. If it is absent, move to the
state-feedback or capability branch instead of inventing another loss.
