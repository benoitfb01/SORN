# Python 2 versus Python 3 core validation

The differential harness executes the untouched Python 2 source from Git
commit `cdad55d55f39e04f568ca1bc0c6036bec8db08fb` and the current Python 3
working tree with identical initial state, inputs, and supplied noise. It then
compares every recorded timestep.

## What is compared

Three 30-step trajectories cover:

- recurrent excitation and inhibition;
- external input and input gain;
- intrinsic plasticity (IP);
- recurrent and input spike-timing-dependent plasticity (STDP);
- inhibitory STDP;
- synaptic scaling and pruning;
- deterministic, noisy, plastic, and frozen-plasticity operation.

A separate controlled case verifies sparse structural-plasticity insertions.
Each trajectory records `x`, `y`, `R_x`, `R_y`, `T_e`, `T_i`, `W_ee`, `W_ei`,
`W_ie`, and `W_eu`, including the initial state. The harness also fails if its
plastic cases do not actually change the neuronal states, thresholds, and
plastic weight matrices, or if its frozen case changes plastic parameters.

Binary neuronal states (`x` and `y`) must be exactly equal. Floating-point
arrays must satisfy `rtol=1e-12` and `atol=1e-12`. The comparator reports the
first mismatching array and index.

## Validated result

The reference environment was Python 2.7.18, NumPy 1.16.6, and SciPy 1.2.3
(x86-64 under Rosetta). The port environment was Python 3.12.13, NumPy 1.26.4,
and SciPy 1.17.1 (arm64).

All 31 arrays passed. The six neuronal-state trajectories were bit-for-bit
identical. The largest difference in any floating-point value was
`2.220446049250313e-16`, roughly machine precision and over four thousand
times smaller than the configured absolute tolerance. Structural-plasticity
weights were identical.

Run the installed harness with:

```sh
sh differential/run_harness.sh
```

The machine-readable report is written to
`artifacts/differential/results/report.json`.

## Isolation and scope

`run_harness.sh` creates the Python 2 source tree with `git archive` from the
pinned reference commit above; it does not run the ported files under Python 2.
The local end-of-life Python 2
runtime, both `.npz` trajectories, the comparison report, and all generated
data live below `artifacts/`, which is ignored by Git.

This result is strong evidence that the exercised core update equations were
preserved. It is not a mathematical guarantee for every parameter combination,
nor does it establish publication-scale equivalence of Figures 1--7. Those
claims require additional long-run, seed-matched statistical comparisons of
the figure observables.
