# criticality — tumbleweed avalanche analysis of the Fig2 N=200 data

Runs the [criticality_tumbleweed](https://github.com/seacore219/criticality_tumbleweed)
avalanche/criticality suite on the 50 N=200 SORN runs used to produce
Del Papa et al. (2017) Fig. 2A–C, i.e. the data under
`artifacts/author_python2/fig2/data/N200/<1..50>/common/result.h5`.

## Provenance

- `analysis.py` is adapted from `analysis.py` in
  [seacore219/SORN_delpapa](https://github.com/seacore219/SORN_delpapa)
  at commit `c6fed43fe288` (the non-delpapa avalanche-analysis code in that
  fork).
- It depends on
  [seacore219/criticality_tumbleweed](https://github.com/seacore219/criticality_tumbleweed),
  installed via `pyproject.toml` as a git dependency pinned to commit
  `05535e8607269531ce0ef0ce0f8c79516250a1a4`.

## Changes from upstream `analysis.py`

Kept as close to upstream as possible; the only changes are what the local
dataset requires:

1. **Folder discovery**: upstream walks an NRP-cluster h_ip sweep
   (`nrp-sweep-data/<batch>_h_ip_X/h_ip_X_runY/test_single/<timestamp>/...`)
   and picks the newest usable retry attempt per run. Here the layout is flat
   (`N200/<run>/common/result.h5`, no retries), so `find_latest_valid_result`
   became `find_run_result` with the same result.h5 validity check.
2. **Analysis window**: `[2000000, 5000000]` instead of upstream's
   `[2500000, 6000000]`, matching Fig2.py (2e6-step transient discarded,
   3e6 stable plastic steps of the 5e6-step runs).
3. **`USE_SPIKES_RASTER = False`** (new flag in `load_sim`): the Fig2
   result.h5 files store only the last 1000 steps of `Spikes`, which doesn't
   cover the analysis window; upstream would have silently analyzed that
   1000-step overlap. All metrics therefore use the full-length population
   activity (`activity * N_e`) — the same signal Fig2.py analyzes. The
   raster-only per-run metrics (branchparam, susceptibility, fano factor,
   activity, cv) are consequently empty in the per-run CSV.
   With `USE_SPIKES_RASTER = True`, `load_sim` requires the raster to
   cover the exact analysis window that `activity` is analyzed on, and
   exits with a clean error otherwise (instead of upstream's silent
   warning). Same safeguard as `analysis_fixed.py` / `analysis_fixed_fits.py`.
4. **Output location**: `criticality/output_<threshold mode>/` (e.g.
   `output_perc/`, `output_adaptive/`) instead of under the sweep root, so
   runs with different avalanche thresholds don't overwrite each other.

One addition on top of upstream: `SIZE_FIT_RANGE` / `DUR_FIT_RANGE` add a
second, fixed-window pooled fit (`fit_mode="manual"` row in the batch CSV)
next to `AV_analysis`'s automatic xmin/xmax search (`fit_mode="auto"`).
It uses the same discrete power-law MLE as the library's `tplfit` and the
same DCC computation; only the fit window is hand-chosen. Defaults are the
Fig2 paper windows (sizes 10–1500, durations 6–60); set either to `None`
to skip it.

Other analysis parameters (`AV_analysis` settings, branching-ratio
`k_max=50` with `exp`/`complex` fits, DFA scales) are upstream's defaults,
unchanged. The avalanche threshold (`AV_THRESHOLD_MODE`) spans two
conventions: upstream's default `"perc"` with `AV_PERC = 0.5` thresholds
at the *median* of population activity (θ ≈ 20 here), while `"adaptive"`
with `ADAPTIVE_THRESHOLD_FRAC = 0.5` thresholds at half the *mean*
(θ = 10 here) — Del Papa's Fig2 convention (on this data the per-run
adaptive thresholds equal Del Papa's single pooled θ exactly). Each mode
writes to its own `output_<mode>/` folder.

## Usage

From the repo root:

```bash
uv run python criticality/analysis.py
```

Outputs (in `criticality/output_<threshold mode>/`):

- `criticality_summary_pooled_by_batch.csv` — one row for N200: pooled
  avalanche fit (α size exponent, β duration exponent, DCC, fit ranges).
  Avalanches are detected per run, then the size/duration events are pooled
  across the 50 runs into a single `AV_analysis` fit.
- `criticality_summary_per_run.csv` — per-run branching ratio
  (mrestimator, exp and complex fits), DFA exponent, n_avalanches,
  threshold used.
- `av_plots/` — pooled avalanche size/duration distribution + scaling plot.
