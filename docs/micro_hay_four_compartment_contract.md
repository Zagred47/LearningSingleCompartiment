# Four-compartment Hay reduction contract

## Scope

This teacher is the smallest spatial reduction used in the project. It is not
the complete Hay morphology and it is not a single isopotential soma. Its four
electrical compartments are:

```text
basal -- soma -- apical trunk -- distal tuft
```

The reduction preserves two distinct input paths, axial propagation, a
proximal-to-distal apical path and the distal calcium/NMDA regime. The axon is
deliberately deferred to a later controlled extension.

## Biophysics retained

- Soma: passive, NaTa_t, Nap_Et2, K_Tst, K_Pst, SKv3_1, SK_E2, Ih,
  Ca_LVAst, Ca_HVA and CaDynamics_E2.
- Basal: passive and Ih, matching `L5PCbiophys3.hoc`.
- Trunk and tuft: passive, NaTa_t, SKv3_1, SK_E2, Ih, Im, Ca_LVAst,
  Ca_HVA and CaDynamics_E2.
- Apical Ih follows the original exponential distance rule.
- The tuft representative lies in the original 685--885 um calcium hot zone;
  the trunk representative lies outside it.
- Somatic calcium decay is 460 ms; apical calcium decay is 122 ms.
- All gates are explicit. In particular, the roughly 2.1 s Nap inactivation,
  slow K_Pst inactivation, Ih, Im and calcium states are not removed.

The state vector has 61 coordinates: 43 intrinsic coordinates and 18
synaptic rise/decay coordinates.

## Synaptic contract

The main dataset has no IClamp or injected-current channel. A fixed bank of 16
excitatory and 8 inhibitory synapses is allocated across basal, trunk and tuft
using represented cable length. Each input is binary at every 0.1 ms bin and
has stable metadata `(synapse_id, type, region)`.

An excitatory event drives both AMPA and NMDA at the same location. An
inhibitory event drives GABA-A. Physical receptor states use the canonical
double-exponential time constants:

| receptor | rise | decay | peak |
|---|---:|---:|---:|
| AMPA | 0.3 ms | 3 ms | 0.4 nS |
| NMDA | 2 ms | 70 ms | 0.4 nS |
| GABA-A | 0.2 ms | 8 ms | 1 nS |

NMDA also retains its voltage-dependent magnesium block. Multiple simultaneous
binary synapse events are summed only inside the physical teacher; the raw
synapse-by-time matrix remains available to the learner.

## Geometry caveat

Electrical cylinder geometry and represented dendritic length are separate.
The former determines membrane area, capacitance and axial resistance; the
latter determines how many virtual synapses represent each collapsed region.
This prevents the reduced cylinders from acquiring the full tree's membrane
area while retaining the original length-weighted input principle.

The four-cylinder geometry is a declared reduction parameter, not a hidden
claim of exact morphology equivalence. A future morphology-reduction audit can
replace these values without changing the state or dataset schema.

## Dataset coverage

Every trajectory receives shuffled blocks from all nine regimes: quiet,
sparse, balanced asynchronous, inhibitory dominant, excitatory dominant,
basal burst, trunk burst, tuft NMDA burst and globally correlated burst. Rates
are smoothed and modulated rather than changed as perfectly sharp constants.

Default data comprise 20 train, 4 validation and 6 test trajectories, each
with 2 s warm-up and 5 s retained data. Seeds are disjoint by split. The HDF5
cache stores binary inputs, complete states, currents, aggregate event counts,
instantaneous rates, regimes and somatic spike labels plus a schema manifest.

## Learning boundary

Only the binary presynaptic spike matrix is a model input. Complete states,
currents, conductances, regime labels and output spikes are targets or
diagnostics. Giving any teacher state back to an input-only surrogate would be
a separate privileged-state experiment and must be labelled as such.
