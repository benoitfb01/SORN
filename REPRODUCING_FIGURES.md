# Reproducing Figures 1--7

This repository contains a Python 3 reproduction runner for Figures 1--7 of
Del Papa, Priesemann, and Triesch (2017), *Criticality meets learning*.

## Data isolation

All generated simulation data and figures are written below:

```text
artifacts/figures_1_7/<profile>/
├── data/       # compressed simulation caches
└── figures/    # PDF and PNG renderings
```

The entire `artifacts/` tree, along with legacy `backup/`, HDF5, pickle, and
rendered figure files, is ignored by Git. Before committing, verify with:

```bash
git status --short
git check-ignore -v artifacts/figures_1_7/validation/data/baseline_n200_r0.npz
```

## Environment

```bash
UV_CACHE_DIR=/private/tmp/sorn-uv-cache uv sync --python 3.12
```

The resolved dependencies are recorded in `uv.lock`.

## Run

Fast end-to-end validation:

```bash
MPLCONFIGDIR=/private/tmp/sorn-mpl-cache \
XDG_CACHE_HOME=/private/tmp/sorn-xdg-cache \
uv run python -m reproduction.run --profile validation
```

Longer local validation:

```bash
MPLCONFIGDIR=/private/tmp/sorn-mpl-cache \
XDG_CACHE_HOME=/private/tmp/sorn-xdg-cache \
uv run python -m reproduction.run --profile practical
```

Publication-scale Figure 2 simulation queue (also supplies Figures 1 and 3):

```bash
uv run python -m reproduction.paper_batch run-figure2 --workers 7
uv run python -m reproduction.paper_batch status
uv run python -m reproduction.paper_batch fit-figure2
```

This queue runs 50 independent replicates at each of `N_E = 50, 100, 200,
400, 800`. Every replicate contains the original two-million-step transient
and three-million-step analysis period. Completed replicates are atomic cache
files, so rerunning the command skips them.

`fit-figure2` reproduces and records the bounded power-law fits, truncated
power-law fits and cutoff parameters, exponential and stretched-exponential
comparisons, theoretical exponent ratio, measured log-log size-versus-duration
slope, and the power-law scale range for every network size. Numerical results
are saved to `paper/analysis/Fig2_fits.json`; the raw pooled avalanche samples
used by the fits are retained alongside it.

Publication protocols for the remaining figures are independently resumable:

```bash
uv run python -m reproduction.paper_batch run-figure4 --workers 7
uv run python -m reproduction.paper_batch analyze-figure4
uv run python -m reproduction.paper_batch run-figure5 --workers 7
uv run python -m reproduction.paper_batch analyze-figure5
uv run python -m reproduction.paper_batch run-figure6 --workers 7
uv run python -m reproduction.paper_batch analyze-figure6
uv run python -m reproduction.paper_batch run-figure7 --workers 7
uv run python -m reproduction.paper_batch analyze-figure7
```

To resume Figures 5--7 and automatically advance through each simulation,
analysis, and rendering stage:

```bash
uv run python -m reproduction.paper_batch run-figures5-7 --workers 7
```

Use `python -m reproduction.paper_batch status` at any time. Publication and
validation caches use different directories and are reported separately.

The runner caches each condition independently and resumes without replacing
completed datasets. Pass `--force` only when the cached data should be
regenerated.

## Scientific status

The `validation` profile is an integration test, not a statistical replication
of the published results. It uses 20,000 transient steps, 30,000 analysis
steps, and one replicate per condition. It exercises every condition and
renders all seven figures.

The original plotting scripts request up to 5,000,000 steps per trajectory,
50 independent trials for the main avalanche distributions, five network
sizes, 50 frozen-plasticity trials per condition, and up to 250 trials for the
input-onset analysis. On the current machine, a single 5,000,000-step
`N_E=200` trajectory takes approximately 12--16 minutes; `N_E=800` is much
slower. A publication-scale replication therefore needs a parallel batch or
cluster run and should not be confused with the local validation artifacts.
