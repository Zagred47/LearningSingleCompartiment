# Capability bench

The bench prevents full-teacher runs from becoming architecture popularity contests. It uses synthetic causal microtasks, never Hay test data.

| ID | Microtask | Required behavior | Main diagnostic |
|---|---|---|---|
| CB-01A | leaky integration | retain slow evidence over gaps | error versus delay and memory spectrum |
| CB-01B | threshold/hysteresis | narrow state-dependent transition | peak/timing error and false-positive spill |
| CB-01C | fast-slow switching | combine slow condition and rapid input | joint-condition generalization |
| CB-02 | four-node propagation | local updates and edge transmission | distance/lag-conditioned NRMSE |

Standard GRU/LSTM, TCN, S4, CfC/LTC and message passing are compared where applicable. Parameter count, data, updates and causal information are matched. Failure prunes the architecture from the corresponding full-teacher branch. Passing authorizes only a one-seed preflight, not a superiority claim.

Generators, splits and seeds must be archived and must require compositional generalization rather than fixed-waveform memorization.

