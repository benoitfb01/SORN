#!/usr/bin/env python3
"""Resumable publication-scale jobs and fitted analyses for Figures 1--3.

The Figure 2 dataset consists of 50 independent five-million-step runs at
each of five network sizes.  Each job writes one atomic, Git-ignored cache so
an interrupted batch can resume without repeating completed replicates.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

# Each simulation is an independent process. Prevent Accelerate/OpenMP from
# creating another pool inside every worker.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/sorn-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/sorn-xdg-cache")

import numpy as np
import powerlaw

from common.sources import NoSource
from common.sorn import Sorn
from reproduction import run as reproduction


NETWORK_SIZES = (50, 100, 200, 400, 800)
REPLICATES = 50
TRANSIENT = 2_000_000
STABLE = 3_000_000
TOTAL_STEPS = TRANSIENT + STABLE
FIT_RANGES = {
    50: {"duration": (2, 10), "size": (3, 100)},
    100: {"duration": (4, 30), "size": (4, 200)},
    200: {"duration": (6, 60), "size": (10, 1500)},
    400: {"duration": (8, 100), "size": (30, 3000)},
    800: {"duration": (10, 150), "size": (70, 7000)},
}
PUBLISHED_FIG2 = {
    "duration_power_law_alpha": 1.45,
    "duration_truncated_alpha": 1.16,
    "duration_truncated_beta": 0.009,
    "size_power_law_alpha": 1.28,
    "size_truncated_alpha": 1.19,
    "size_truncated_beta": 0.001,
    "theoretical_exponent_ratio": 1.6,
    "comparison_gamma": 1.3,
}
FIG4_CONDITIONS = ("all", "random", "ip", "all_but_ip")
FIG4_STEPS = 2_000_000
FIG5_GAUSSIAN_REPLICATES = {0.005: 2, 0.05: 2, 5.0: 2}
FIG5_SPIKE_REPLICATES = {0.0: 65, 0.05: 30, 0.10: 30}
FIG6_REPLICATES = 250
FIG6_SECTION_STEPS = 2_000_000
FIG7_REPLICATES = 2
FIG7_COUNTING_N = (4, 6, 8, 14, 20)
FIG7_COUNTING_VARIANTS = ("noise", "no_noise", "original")
FIG7_RANDOM_LENGTHS = (10, 20, 50, 100)
FIG7_PLASTIC_STEPS = 5_000
FIG7_TRAIN_STEPS = 5_000
FIG7_TEST_STEPS = 5_000
FIG7_AVALANCHE_STEPS = 2_000_000

PAPER_ROOT = reproduction.ARTIFACTS / "paper"
DATA = PAPER_ROOT / "data"
ANALYSIS = PAPER_ROOT / "analysis"


def cache_path(network_size: int, replicate: int) -> Path:
    return DATA / f"baseline_n{network_size}_r{replicate}.npz"


def fig4_cache_path(condition: str, replicate: int) -> Path:
    return DATA / f"fig4_{condition}_r{replicate}.npz"


def number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def fig5_name(kind: str, value: float, replicate: int) -> str:
    return f"fig5_{kind}_{number_label(value)}_r{replicate}"


def fig5_cache_path(kind: str, value: float, replicate: int) -> Path:
    return DATA / f"{fig5_name(kind, value, replicate)}.npz"


def fig6_cache_path(replicate: int) -> Path:
    return DATA / f"fig6_input_r{replicate}.npz"


def fig7_counting_cache_path(variant: str, n: int, replicate: int) -> Path:
    return DATA / f"fig7_counting_{variant}_n{n}_r{replicate}.npz"


def fig7_random_cache_path(length: int, replicate: int) -> Path:
    return DATA / f"fig7_random_l{length}_r{replicate}.npz"


def seed_for(network_size: int, replicate: int) -> int:
    return 1000 + network_size * 10 + replicate


def run_baseline(network_size: int, replicate: int, force: bool) -> None:
    if network_size not in NETWORK_SIZES:
        raise SystemExit(f"unsupported network size: {network_size}")
    if not 0 <= replicate < REPLICATES:
        raise SystemExit(f"replicate must be between 0 and {REPLICATES - 1}")
    DATA.mkdir(parents=True, exist_ok=True)
    reproduction.DATA = DATA
    name = f"baseline_n{network_size}_r{replicate}"
    result = reproduction.simulate(
        name,
        reproduction.config(n_e=network_size),
        NoSource(),
        TOTAL_STEPS,
        seed_for(network_size, replicate),
        force=force,
        progress_every=250_000,
    )
    print(f"completed {name} in {float(result['elapsed_seconds']) / 60:.2f} min",
          flush=True)


def completed_jobs() -> dict[int, list[int]]:
    return {
        network_size: [
            replicate for replicate in range(REPLICATES)
            if cache_path(network_size, replicate).exists()
        ]
        for network_size in NETWORK_SIZES
    }


def print_status() -> None:
    completed = completed_jobs()
    total = sum(len(value) for value in completed.values())
    for network_size in NETWORK_SIZES:
        print(f"N_E={network_size}: {len(completed[network_size])}/{REPLICATES}")
    print(f"Total: {total}/{len(NETWORK_SIZES) * REPLICATES}")
    fig4_total = 0
    for condition in FIG4_CONDITIONS:
        count = sum(fig4_cache_path(condition, replicate).exists()
                    for replicate in range(REPLICATES))
        fig4_total += count
        print(f"Figure 4 {condition}: {count}/{REPLICATES}")
    print(f"Figure 4 total: {fig4_total}/{len(FIG4_CONDITIONS) * REPLICATES}")
    fig5_total = 0
    fig5_expected = 0
    for kind, design in (("gaussian", FIG5_GAUSSIAN_REPLICATES),
                         ("spikes", FIG5_SPIKE_REPLICATES)):
        for value, expected in design.items():
            count = sum(fig5_cache_path(kind, value, replicate).exists()
                        for replicate in range(expected))
            fig5_total += count
            fig5_expected += expected
            print(f"Figure 5 {kind}={value:g}: {count}/{expected}")
    print(f"Figure 5 total: {fig5_total}/{fig5_expected}")
    fig6_count = sum(fig6_cache_path(replicate).exists()
                     for replicate in range(FIG6_REPLICATES))
    print(f"Figure 6 input phases: {fig6_count}/{FIG6_REPLICATES}")
    fig7_counting = sum(
        fig7_counting_cache_path(variant, n, replicate).exists()
        for variant in FIG7_COUNTING_VARIANTS for n in FIG7_COUNTING_N
        for replicate in range(FIG7_REPLICATES)
    )
    fig7_random = sum(
        fig7_random_cache_path(length, replicate).exists()
        for length in FIG7_RANDOM_LENGTHS
        for replicate in range(FIG7_REPLICATES)
    )
    print(f"Figure 7 counting: {fig7_counting}/30")
    print(f"Figure 7 random: {fig7_random}/8")


def write_manifest() -> Path:
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Del Papa et al. (2017), Figures 1--3 baseline / Figure 2",
        "network_sizes": list(NETWORK_SIZES),
        "replicates_per_network_size": REPLICATES,
        "transient_steps": TRANSIENT,
        "stable_steps": STABLE,
        "total_steps_per_replicate": TOTAL_STEPS,
        "seed_formula": "1000 + 10 * N_e + zero_based_replicate",
        "fit_ranges": FIT_RANGES,
        "published_figure2_values": PUBLISHED_FIG2,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "powerlaw": importlib.metadata.version("powerlaw"),
        },
        "data_location": str(DATA),
        "git_ignored": True,
    }
    path = PAPER_ROOT / "manifest.json"
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    print(f"wrote {path}")
    return path


def _baseline_job(job: tuple[int, int]) -> tuple[int, int]:
    network_size, replicate = job
    run_baseline(network_size, replicate, False)
    return job


class NullStats:
    def add(self) -> None:
        pass


def run_figure4_condition(condition: str, replicate: int,
                          force: bool = False) -> None:
    if condition not in FIG4_CONDITIONS:
        raise SystemExit(f"unsupported Figure 4 condition: {condition}")
    path = fig4_cache_path(condition, replicate)
    if path.exists() and not force:
        return
    DATA.mkdir(parents=True, exist_ok=True)
    condition_index = FIG4_CONDITIONS.index(condition)
    seed = 40_000 + condition_index * 1_000 + replicate
    reproduction.seed_all(seed)
    cfg = reproduction.config()
    sorn = Sorn(cfg, NoSource())
    sorn.stats = NullStats()
    started = time.perf_counter()
    if condition != "random":
        sorn.simulation(FIG4_STEPS)
    reproduction.freeze(cfg, "all" if condition == "random" else condition)
    recorder = reproduction.Recorder(
        sorn, FIG4_STEPS, label=f"fig4_{condition}_r{replicate}",
        progress_every=250_000,
    )
    sorn.stats = recorder
    sorn.simulation(FIG4_STEPS)
    result = {
        "activity": recorder.activity,
        "connection_fraction": recorder.connection_fraction,
        "spikes": recorder.spikes,
        "elapsed_seconds": np.array(time.perf_counter() - started),
        "n_e": np.array(cfg.N_e),
        "training_steps": np.array(0 if condition == "random" else FIG4_STEPS),
        "analysis_steps": np.array(FIG4_STEPS),
        "seed": np.array(seed),
        "condition": np.array(condition),
    }
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(path)
    print(f"completed fig4_{condition}_r{replicate} in "
          f"{float(result['elapsed_seconds']) / 60:.2f} min", flush=True)


def _figure4_job(job: tuple[str, int]) -> tuple[str, int]:
    condition, replicate = job
    run_figure4_condition(condition, replicate)
    return job


def run_figure4_batch(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = [(condition, replicate) for condition in FIG4_CONDITIONS
            for replicate in range(REPLICATES)
            if not fig4_cache_path(condition, replicate).exists()]
    print(f"Starting {len(jobs)} pending Figure 4 jobs with {workers} workers",
          flush=True)
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_figure4_job, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                future.result()
            except Exception:
                print(f"FAILED Figure 4 {job[0]} replicate={job[1]}", flush=True)
                raise
            print(f"Figure 4 batch progress: {completed}/{len(jobs)}", flush=True)


def run_figure5_condition(kind: str, value: float, replicate: int,
                          force: bool = False) -> None:
    designs = {"gaussian": FIG5_GAUSSIAN_REPLICATES,
               "spikes": FIG5_SPIKE_REPLICATES}
    if kind not in designs or value not in designs[kind]:
        raise SystemExit(f"unsupported Figure 5 condition: {kind}={value}")
    if not 0 <= replicate < designs[kind][value]:
        raise SystemExit("Figure 5 replicate is outside the published design")
    reproduction.DATA = DATA
    kind_offset = 0 if kind == "gaussian" else 10_000
    value_offset = list(designs[kind]).index(value) * 1_000
    cfg = (reproduction.config(noise_var=value) if kind == "gaussian" else
           reproduction.config(noise_var=0.0, spike_noise=value))
    result = reproduction.simulate(
        fig5_name(kind, value, replicate), cfg, NoSource(), TOTAL_STEPS,
        50_000 + kind_offset + value_offset + replicate,
        force=force, progress_every=250_000,
    )
    print(f"completed {fig5_name(kind, value, replicate)} in "
          f"{float(result['elapsed_seconds']) / 60:.2f} min", flush=True)


def _figure5_job(job: tuple[str, float, int]) -> tuple[str, float, int]:
    run_figure5_condition(*job)
    return job


def run_figure5_batch(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = []
    for kind, design in (("gaussian", FIG5_GAUSSIAN_REPLICATES),
                         ("spikes", FIG5_SPIKE_REPLICATES)):
        jobs.extend((kind, value, replicate)
                    for value, count in design.items()
                    for replicate in range(count)
                    if not fig5_cache_path(kind, value, replicate).exists())
    print(f"Starting {len(jobs)} pending Figure 5 jobs with {workers} workers",
          flush=True)
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_figure5_job, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                future.result()
            except Exception:
                print(f"FAILED Figure 5 {job}", flush=True)
                raise
            print(f"Figure 5 batch progress: {completed}/{len(jobs)}", flush=True)


def run_figure6_replicate(replicate: int, force: bool = False) -> None:
    if not 0 <= replicate < FIG6_REPLICATES:
        raise SystemExit("Figure 6 replicate is outside the published design")
    path = fig6_cache_path(replicate)
    if path.exists() and not force:
        return
    DATA.mkdir(parents=True, exist_ok=True)
    seed = 60_000 + replicate
    reproduction.seed_all(seed)
    cfg = reproduction.config(input_fraction=0.02)
    cfg.input_gain = 100_000_000
    sorn = Sorn(cfg, NoSource(N_i=cfg.N_u_e))
    sorn.stats = NullStats()
    started = time.perf_counter()
    sorn.simulation(FIG6_SECTION_STEPS)

    normal = reproduction.Recorder(
        sorn, FIG6_SECTION_STEPS, label=f"fig6_normal_r{replicate}",
        progress_every=250_000,
    )
    sorn.stats = normal
    sorn.simulation(FIG6_SECTION_STEPS)

    words = list("ABCDEFGHIJ")
    source = reproduction.CountingSource(
        words, np.ones((10, 10)) / 10.0, cfg.N_u_e, 0, avoid=False
    )
    sorn.source = source
    sorn.W_eu = source.generate_connection_e(cfg.N_e)
    input_phase = reproduction.Recorder(
        sorn, FIG6_SECTION_STEPS, label=f"fig6_input_r{replicate}",
        progress_every=250_000,
    )
    sorn.stats = input_phase
    sorn.simulation(FIG6_SECTION_STEPS)

    result = {
        "normal_activity": normal.activity,
        "input_activity": input_phase.activity,
        "elapsed_seconds": np.array(time.perf_counter() - started),
        "n_e": np.array(cfg.N_e),
        "section_steps": np.array(FIG6_SECTION_STEPS),
        "seed": np.array(seed),
    }
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(path)
    print(f"completed fig6_input_r{replicate} in "
          f"{float(result['elapsed_seconds']) / 60:.2f} min", flush=True)


def _figure6_job(replicate: int) -> int:
    run_figure6_replicate(replicate)
    return replicate


def run_figure6_batch(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = [replicate for replicate in range(FIG6_REPLICATES)
            if not fig6_cache_path(replicate).exists()]
    print(f"Starting {len(jobs)} pending Figure 6 jobs with {workers} workers",
          flush=True)
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_figure6_job, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            replicate = futures[future]
            try:
                future.result()
            except Exception:
                print(f"FAILED Figure 6 replicate={replicate}", flush=True)
                raise
            print(f"Figure 6 batch progress: {completed}/{len(jobs)}", flush=True)


def restore_plasticity(cfg, *, noise_variance: float) -> None:
    cfg.W_ee.eta_stdp = 0.004
    cfg.W_ei.eta_istdp = 0.001
    cfg.W_ee.sp_prob = cfg.N_e * (cfg.N_e - 1) * (0.1 / (200 * 199))
    cfg.eta_ip = 0.01
    cfg.noise_sig = np.sqrt(noise_variance)


def run_figure7_counting(variant: str, n: int, replicate: int,
                         force: bool = False) -> None:
    if variant not in FIG7_COUNTING_VARIANTS or n not in FIG7_COUNTING_N:
        raise SystemExit("unsupported Figure 7 counting condition")
    path = fig7_counting_cache_path(variant, n, replicate)
    if path.exists() and not force:
        return
    DATA.mkdir(parents=True, exist_ok=True)
    seed = 70_000 + FIG7_COUNTING_VARIANTS.index(variant) * 1_000 + n * 10 + replicate
    reproduction.seed_all(seed)
    noise_variance = 0.0 if variant == "no_noise" else 0.05
    cfg = reproduction.config(noise_var=noise_variance, input_fraction=0.05)
    if variant == "original":
        cfg.W_ee.sp_prob = 0.0
        cfg.W_ee.sp_initial = 0.0
        cfg.W_ei.lamb = cfg.N_e
        cfg.W_ei.eta_istdp = 0.0
    words = ["A" + "B" * n + "C", "D" + "E" * n + "F"]
    source = reproduction.CountingSource(
        words, np.ones((2, 2)) / 2.0, cfg.N_u_e, 0, avoid=False
    )
    sorn = Sorn(cfg, source)
    sorn.stats = NullStats()
    started = time.perf_counter()
    sorn.simulation(FIG7_PLASTIC_STEPS)
    reproduction.freeze(cfg, "all")
    readout = reproduction.Recorder(
        sorn, FIG7_TRAIN_STEPS + FIG7_TEST_STEPS,
        task_steps=FIG7_TRAIN_STEPS + FIG7_TEST_STEPS,
    )
    sorn.stats = readout
    sorn.simulation(FIG7_TRAIN_STEPS + FIG7_TEST_STEPS)

    avalanche_activity = np.empty(0, dtype=np.uint16)
    if variant == "noise" and n in (4, 20):
        restore_plasticity(cfg, noise_variance=0.0)
        avalanche = reproduction.Recorder(
            sorn, FIG7_AVALANCHE_STEPS,
            label=f"fig7_counting_n{n}_r{replicate}", progress_every=250_000,
        )
        sorn.stats = avalanche
        sorn.simulation(FIG7_AVALANCHE_STEPS)
        avalanche_activity = avalanche.activity
    result = {
        "states": readout.task_states,
        "letters": readout.letters,
        "avalanche_activity": avalanche_activity,
        "elapsed_seconds": np.array(time.perf_counter() - started),
        "seed": np.array(seed), "n": np.array(n), "variant": np.array(variant),
    }
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(path)
    print(f"completed fig7 counting {variant} n={n} r={replicate}", flush=True)


def run_figure7_random(length: int, replicate: int, force: bool = False) -> None:
    if length not in FIG7_RANDOM_LENGTHS:
        raise SystemExit("unsupported Figure 7 random-task length")
    path = fig7_random_cache_path(length, replicate)
    if path.exists() and not force:
        return
    DATA.mkdir(parents=True, exist_ok=True)
    seed = 80_000 + length * 10 + replicate
    reproduction.seed_all(seed)
    cfg = reproduction.config(noise_var=0.05, input_fraction=0.05)
    rng = random.Random(seed)
    word = "".join(rng.choice("ABCDEFGHIJ") for unused in range(length))
    source = reproduction.CountingSource(
        [word, word], np.ones((2, 2)) / 2.0, cfg.N_u_e, 0, avoid=False
    )
    sorn = Sorn(cfg, source)
    sorn.stats = NullStats()
    started = time.perf_counter()
    sorn.simulation(FIG7_PLASTIC_STEPS)
    reproduction.freeze(cfg, "all")
    readout = reproduction.Recorder(
        sorn, FIG7_TRAIN_STEPS + FIG7_TEST_STEPS,
        task_steps=FIG7_TRAIN_STEPS + FIG7_TEST_STEPS,
    )
    sorn.stats = readout
    sorn.simulation(FIG7_TRAIN_STEPS + FIG7_TEST_STEPS)
    restore_plasticity(cfg, noise_variance=0.05)
    avalanche = reproduction.Recorder(
        sorn, FIG7_AVALANCHE_STEPS,
        label=f"fig7_random_l{length}_r{replicate}", progress_every=250_000,
    )
    sorn.stats = avalanche
    sorn.simulation(FIG7_AVALANCHE_STEPS)
    result = {
        "states": readout.task_states,
        "letters": readout.letters,
        "avalanche_activity": avalanche.activity,
        "elapsed_seconds": np.array(time.perf_counter() - started),
        "seed": np.array(seed), "length": np.array(length),
    }
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(path)
    print(f"completed fig7 random L={length} r={replicate}", flush=True)


def _figure7_job(job: tuple) -> tuple:
    if job[0] == "counting":
        run_figure7_counting(*job[1:])
    else:
        run_figure7_random(*job[1:])
    return job


def run_figure7_batch(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = [
        ("counting", variant, n, replicate)
        for variant in FIG7_COUNTING_VARIANTS for n in FIG7_COUNTING_N
        for replicate in range(FIG7_REPLICATES)
        if not fig7_counting_cache_path(variant, n, replicate).exists()
    ]
    jobs.extend(
        ("random", length, replicate)
        for length in FIG7_RANDOM_LENGTHS for replicate in range(FIG7_REPLICATES)
        if not fig7_random_cache_path(length, replicate).exists()
    )
    print(f"Starting {len(jobs)} pending Figure 7 jobs with {workers} workers",
          flush=True)
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_figure7_job, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                future.result()
            except Exception:
                print(f"FAILED Figure 7 {job}", flush=True)
                raise
            print(f"Figure 7 batch progress: {completed}/{len(jobs)}", flush=True)


def run_figures5_through7(workers: int) -> None:
    """Run every remaining stage in order, resuming completed atomic caches."""
    run_figure5_batch(workers)
    analyze_and_render_figure5()
    run_figure6_batch(workers)
    analyze_and_render_figure6()
    run_figure7_batch(workers)
    analyze_and_render_figure7()


def run_figure2_batch(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    write_manifest()
    # Finish N=200 first so panels A--C and their fits become available early;
    # schedule the longest remaining runs before the short ones to reduce tail.
    priority = (200, 800, 400, 100, 50)
    jobs = [(network_size, replicate) for network_size in priority
            for replicate in range(REPLICATES)
            if not cache_path(network_size, replicate).exists()]
    print(f"Starting {len(jobs)} pending Figure 2 jobs with {workers} workers",
          flush=True)
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_baseline_job, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                future.result()
            except Exception:
                print(f"FAILED N_E={job[0]} replicate={job[1]}", flush=True)
                raise
            print(f"batch progress: {completed}/{len(jobs)} newly completed",
                  flush=True)


def load_avalanches(network_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    missing = [r for r in range(REPLICATES)
               if not cache_path(network_size, r).exists()]
    if missing:
        raise SystemExit(
            f"N_E={network_size} is missing {len(missing)} replicates: {missing}"
        )
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    avalanche_path = ANALYSIS / f"fig2_avalanches_n{network_size}.npz"
    if avalanche_path.exists():
        with np.load(avalanche_path) as cached:
            return cached["duration"], cached["size"], int(cached["threshold"])

    activity_sum = 0
    activity_count = 0
    for replicate in range(REPLICATES):
        with np.load(cache_path(network_size, replicate)) as run:
            stable_activity = run["activity"][-STABLE:]
            activity_sum += int(stable_activity.sum(dtype=np.uint64))
            activity_count += stable_activity.size
    threshold = int((activity_sum / activity_count) / 2.0)
    durations: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    for replicate in range(REPLICATES):
        with np.load(cache_path(network_size, replicate)) as run:
            duration, size = reproduction.avalanches(
                run["activity"][-STABLE:], threshold
            )
        durations.append(duration)
        sizes.append(size)
        print(f"avalanches N_E={network_size}: {replicate + 1}/{REPLICATES}",
              flush=True)
    duration = np.concatenate(durations)
    size = np.concatenate(sizes)
    temporary = avalanche_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, duration=duration, size=size,
                        threshold=np.array(threshold))
    temporary.replace(avalanche_path)
    return duration, size, threshold


def power_law_result(data: np.ndarray, xmin: int, xmax: int) -> dict[str, float]:
    fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax, discrete=True, verbose=False)
    return {
        "alpha": float(fit.power_law.alpha),
        "sigma": float(fit.power_law.sigma),
        "xmin": float(fit.xmin),
        "xmax": float(fit.xmax),
        "scale_range_decades": float(np.log10(xmax) - np.log10(xmin)),
        "sample_count": int(np.count_nonzero((data >= xmin) & (data <= xmax))),
    }


def truncated_result(data: np.ndarray, xmin: int) -> tuple[dict[str, float], object]:
    fit = powerlaw.Fit(data, xmin=xmin, discrete=True, verbose=False)
    distribution = fit.truncated_power_law
    result = {
        "alpha": float(distribution.parameter1),
        "beta": float(distribution.parameter2),
        "xmin": float(fit.xmin),
        "sample_count": int(np.count_nonzero(data >= xmin)),
    }
    return result, fit


def comparison(fit, alternative: str) -> dict[str, float]:
    ratio, p_value = fit.distribution_compare(
        "power_law", alternative, normalized_ratio=True
    )
    return {"normalized_loglikelihood_ratio": float(ratio),
            "p_value": float(p_value)}


def empirical_pdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, counts = np.unique(data, return_counts=True)
    return values, counts / counts.sum()


def scaled_model_curve(values: np.ndarray, density: np.ndarray, xmin: int,
                       alpha: float, beta: float = 0.0,
                       xmax: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    upper = int(values.max()) if xmax is None else xmax
    x = np.arange(xmin, upper + 1, dtype=float)
    model = x ** (-alpha) * np.exp(-beta * x)
    anchor = np.flatnonzero(values >= xmin)[0]
    model *= density[anchor] / model[0]
    return x, model


def render_figure2(results: dict[str, object],
                   avalanche_data: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    import matplotlib.pyplot as plt

    figures = PAPER_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(8, 8 / 1.718))
    duration, size = avalanche_data[200]
    n200 = results["network_sizes"]["200"]
    full = results["n200_full_fits"]

    duration_x, duration_y = empirical_pdf(duration)
    axes[0, 0].loglog(duration_x, duration_y, ".", color="gray", ms=2,
                      zorder=1)
    duration_power = n200["duration_power_law"]
    x, y = scaled_model_curve(duration_x, duration_y, 6,
                              duration_power["alpha"], xmax=60)
    axes[0, 0].loglog(x, y, color="#006BB2", lw=2,
                      label=rf"$\alpha={duration_power['alpha']:.2f}$")
    duration_truncated = full["duration_truncated_power_law"]
    x, y = scaled_model_curve(duration_x, duration_y, 6,
                              duration_truncated["alpha"],
                              duration_truncated["beta"])
    axes[0, 0].loglog(
        x, y, color="black", lw=1.5,
        label=(rf"$\alpha^*={duration_truncated['alpha']:.2f},\ "
               rf"\beta^*_\alpha={duration_truncated['beta']:.3f}$"),
    )

    size_x, size_y = empirical_pdf(size)
    axes[0, 1].loglog(size_x, size_y, ".", color="gray", ms=2, zorder=1)
    size_power = n200["size_power_law"]
    x, y = scaled_model_curve(size_x, size_y, 10, size_power["alpha"],
                              xmax=1500)
    axes[0, 1].loglog(x, y, color="#B22400", lw=2,
                      label=rf"$\tau={size_power['alpha']:.2f}$")
    size_truncated = full["size_truncated_power_law"]
    x, y = scaled_model_curve(size_x, size_y, 10, size_truncated["alpha"],
                              size_truncated["beta"])
    axes[0, 1].loglog(
        x, y, color="black", lw=1.5,
        label=(rf"$\tau^*={size_truncated['alpha']:.2f},\ "
               rf"\beta^*_\tau={size_truncated['beta']:.3f}$"),
    )

    counts = np.bincount(duration)
    size_sums = np.bincount(duration, weights=size)
    durations = np.flatnonzero(counts)
    mean_sizes = size_sums[durations] / counts[durations]
    normalized_mean_sizes = mean_sizes / mean_sizes.sum()
    axes[0, 2].loglog(durations, normalized_mean_sizes, ".", color="gray",
                      ms=2, label=rf"$\gamma_{{data}}={full['gamma_data_loglog_slope']:.2f}$")
    reference_x = np.arange(1, int(durations.max()) + 1)
    anchor = normalized_mean_sizes.min()
    gamma = full["theoretical_exponent_ratio"]
    axes[0, 2].loglog(reference_x, anchor * reference_x ** gamma,
                      color="red", lw=1.5,
                      label=rf"$(\alpha-1)/(\tau-1)={gamma:.2f}$")
    axes[0, 2].loglog(reference_x, anchor * reference_x ** 1.3, "--k",
                      lw=1.5, label=r"$\gamma=1.3$")

    colors = ("blue", "green", "red", "c", "m")
    for network_size, color in zip(NETWORK_SIZES, colors):
        durations_n, sizes_n = avalanche_data[network_size]
        x, y = empirical_pdf(durations_n)
        axes[1, 0].loglog(x, y, color=color, lw=1)
        x, y = empirical_pdf(sizes_n)
        axes[1, 1].loglog(x, y, color=color, lw=1, label=str(network_size))
    reference_x = np.arange(1, 10_001)
    axes[1, 0].loglog(reference_x, reference_x ** (-duration_power["alpha"]),
                      "--", color="#006BB2", lw=2)
    reference_x = np.arange(1, 100_001)
    axes[1, 1].loglog(reference_x, reference_x ** (-size_power["alpha"]),
                      "--", color="#B22400", lw=2)

    network = np.asarray(NETWORK_SIZES)
    duration_ranges = [results["network_sizes"][str(n)]["duration_power_law"]
                       ["scale_range_decades"] for n in NETWORK_SIZES]
    size_ranges = [results["network_sizes"][str(n)]["size_power_law"]
                   ["scale_range_decades"] for n in NETWORK_SIZES]
    axes[1, 2].semilogx(network, size_ranges, "-o", color="#B22400",
                        label="size")
    axes[1, 2].semilogx(network, duration_ranges, "-o", color="#006BB2",
                        label="duration")

    labels = ((r"$T$", r"$f(T)$"), (r"$S$", r"$f(S)$"),
              (r"$T$", r"$\langle S\rangle(T)$"),
              (r"$T$", r"$f(T)$"), (r"$S$", r"$f(S)$"),
              ("Network size", "Power-law scale range"))
    for label, ax, (xlabel, ylabel) in zip("ABCDEF", axes.flat, labels):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines[["right", "top"]].set_visible(False)
        ax.text(-0.22, 1.05, label, transform=ax.transAxes,
                fontweight="bold", fontsize=13)
    axes[0, 0].set_xlim(1, 300); axes[0, 0].set_ylim(1e-4, 1)
    axes[0, 1].set_xlim(1, 3_000); axes[0, 1].set_ylim(1e-5, 1e-1)
    axes[0, 2].set_xlim(1, 200); axes[0, 2].set_ylim(1e-6, 1e-1)
    axes[1, 0].set_xlim(1, 10_000); axes[1, 0].set_ylim(1e-6, 1)
    axes[1, 1].set_xlim(1, 100_000); axes[1, 1].set_ylim(1e-6, 1)
    axes[1, 2].set_xlim(20, 2_000); axes[1, 2].set_ylim(0, 3)
    axes[1, 2].set_xticks([100, 1_000])
    axes[1, 2].set_yticks([0, 1, 2, 3])
    axes[0, 0].legend(frameon=False, fontsize=8, title="Fit parameters",
                      loc="upper left")
    axes[0, 1].legend(frameon=False, fontsize=8, title="Fit parameters",
                      loc="upper left")
    axes[0, 2].legend(frameon=False, fontsize=8, title="Exponent ratio",
                      loc="lower right")
    axes[1, 1].legend(frameon=False, title="Network size",
                      loc="upper right")
    axes[1, 2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "Fig2.pdf")
    fig.savefig(figures / "Fig2.png", dpi=300)
    plt.close(fig)


def render_figure2_from_cache() -> None:
    fit_path = ANALYSIS / "Fig2_fits.json"
    if not fit_path.exists():
        raise SystemExit(f"fit report not found: {fit_path}")
    results = json.loads(fit_path.read_text())
    avalanche_data = {}
    for network_size in NETWORK_SIZES:
        path = ANALYSIS / f"fig2_avalanches_n{network_size}.npz"
        with np.load(path) as cached:
            avalanche_data[network_size] = (cached["duration"], cached["size"])
    render_figure2(results, avalanche_data)
    print(f"rendered {PAPER_ROOT / 'figures' / 'Fig2.pdf'}")


def load_run(network_size: int, replicate: int,
             keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(cache_path(network_size, replicate)) as cached:
        return {key: cached[key] for key in keys}


def render_figures1_and3() -> None:
    reproduction.FIGURES = PAPER_ROOT / "figures"
    reproduction.FIGURES.mkdir(parents=True, exist_ok=True)
    profile = reproduction.PROFILES["paper"]
    first = load_run(200, 0, ("activity", "connection_fraction", "spikes"))
    reproduction.plot_fig1(first, profile)
    runs200 = [load_run(200, replicate, ("activity",))
               for replicate in range(REPLICATES)]
    reproduction.plot_fig3(runs200, profile)
    print(f"rendered Figures 1 and 3 in {reproduction.FIGURES}")


def figure4_activity(condition: str, replicate: int) -> np.ndarray:
    if condition == "normal":
        with np.load(cache_path(200, replicate)) as cached:
            return cached["activity"][TRANSIENT:TRANSIENT + FIG4_STEPS]
    with np.load(fig4_cache_path(condition, replicate)) as cached:
        return cached["activity"]


def pooled_threshold(activity_histogram: np.ndarray, sample_count: int,
                     threshold: str) -> int:
    if threshold == "half":
        mean = np.dot(np.arange(activity_histogram.size), activity_histogram)
        return int((mean / sample_count) / 2.0)
    percentile = float(threshold.removesuffix("percent"))
    target = percentile / 100.0 * sample_count
    return int(np.searchsorted(np.cumsum(activity_histogram), target,
                               side="left"))


def figure4_avalanches(condition: str, threshold: str = "half") \
        -> tuple[np.ndarray, np.ndarray, int]:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS / f"fig4_avalanches_{condition}_{threshold}.npz"
    if path.exists():
        with np.load(path) as cached:
            return cached["duration"], cached["size"], int(cached["threshold"])
    histogram = np.zeros(201, dtype=np.int64)
    for replicate in range(REPLICATES):
        activity = figure4_activity(condition, replicate)
        counts = np.bincount(activity, minlength=201)
        histogram[:counts.size] += counts
    sample_count = FIG4_STEPS * REPLICATES
    theta = pooled_threshold(histogram, sample_count, threshold)
    durations, sizes = [], []
    for replicate in range(REPLICATES):
        duration, size = reproduction.avalanches(
            figure4_activity(condition, replicate), theta
        )
        durations.append(duration)
        sizes.append(size)
    duration = np.concatenate(durations)
    size = np.concatenate(sizes)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, duration=duration, size=size,
                        threshold=np.array(theta))
    temporary.replace(path)
    print(f"Figure 4 {condition} {threshold}: theta={theta}, "
          f"avalanches={duration.size:,}", flush=True)
    return duration, size, theta


def shade_log_pdf(ax, low_data: np.ndarray, high_data: np.ndarray,
                  color: str) -> None:
    low_x, low_y = log_binned_pdf(low_data)
    high_x, high_y = log_binned_pdf(high_data)
    if not low_x.size or not high_x.size:
        return
    positive = np.unique(np.concatenate((low_x, high_x)))
    low_interp = np.interp(positive, low_x, low_y, left=np.nan, right=np.nan)
    high_interp = np.interp(positive, high_x, high_y, left=np.nan, right=np.nan)
    bottom = np.fmin(low_interp, high_interp)
    top = np.fmax(low_interp, high_interp)
    valid = np.isfinite(bottom) & np.isfinite(top) & (bottom > 0)
    ax.fill_between(positive[valid], bottom[valid], top[valid], color=color,
                    alpha=0.15, linewidth=0)


def log_binned_pdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data)
    data = data[data > 0]
    if not data.size:
        return np.empty(0), np.empty(0)
    unique = np.unique(data)
    if unique.size == 1:
        return unique.astype(float), np.ones(1)
    edges, probability = powerlaw.pdf(data, linear_bins=False)
    centers = np.sqrt(edges[:-1] * edges[1:])
    valid = np.isfinite(probability) & (probability > 0)
    return centers[valid], probability[valid]


def analyze_and_render_figure4() -> None:
    import matplotlib.pyplot as plt

    all_conditions = ("normal",) + FIG4_CONDITIONS
    results: dict[str, object] = {}
    distributions = {}
    for condition in all_conditions:
        duration, size, theta = figure4_avalanches(condition, "half")
        duration5, size5, theta5 = figure4_avalanches(condition, "5percent")
        duration25, size25, theta25 = figure4_avalanches(condition, "25percent")
        distributions[condition] = {
            "duration": duration, "size": size,
            "duration5": duration5, "size5": size5,
            "duration25": duration25, "size25": size25,
        }
        results[condition] = {
            "replicates": REPLICATES,
            "analysis_steps_per_replicate": FIG4_STEPS,
            "half_mean_threshold": theta,
            "percentile_5_threshold": theta5,
            "percentile_25_threshold": theta25,
            "avalanche_count": int(duration.size),
        }
    summary = ANALYSIS / "Fig4_summary.json"
    temporary = summary.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(summary)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    panels = (
        (axes[0, 0], "duration", (("normal", "black", "SORN"),
                                  ("all", "c", "Frozen (all)"),
                                  ("random", "red", "Random network"))),
        (axes[0, 1], "size", (("normal", "black", "SORN"),
                              ("all", "c", "Frozen (all)"),
                              ("random", "red", "Random network"))),
        (axes[1, 0], "duration", (("ip", "red", "Frozen (IP)"),
                                  ("all_but_ip", "c", "Frozen (all but IP)"))),
        (axes[1, 1], "size", (("ip", "red", "Frozen (IP)"),
                              ("all_but_ip", "c", "Frozen (all but IP)"))),
    )
    for ax, quantity, entries in panels:
        for condition, color, label in entries:
            x, y = log_binned_pdf(distributions[condition][quantity])
            ax.loglog(x, y, color=color, lw=2 if condition == "normal" else 1.25,
                      label=label)
            if condition != "random":
                shade_log_pdf(ax, distributions[condition][quantity + "5"],
                              distributions[condition][quantity + "25"], color)
        ax.set_xlabel(r"$T$" if quantity == "duration" else r"$S$")
        ax.set_ylabel(r"$f(T)$" if quantity == "duration" else r"$f(S)$")
        ax.set_xlim(1, 3_000 if quantity == "duration" else 20_000)
        ax.set_ylim(1e-6 if quantity == "duration" else 1e-7, 1)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    for label, ax in zip("ABCD", axes.flat):
        ax.text(-0.2, 1.02, label, transform=ax.transAxes,
                fontweight="bold", fontsize=12)
    fig.tight_layout()
    figures = PAPER_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "Fig4.pdf")
    fig.savefig(figures / "Fig4.png", dpi=300)
    plt.close(fig)
    print(f"rendered {figures / 'Fig4.pdf'}")


def figure5_activity(kind: str, value: float, replicate: int) -> np.ndarray:
    with np.load(fig5_cache_path(kind, value, replicate)) as cached:
        return cached["activity"][-STABLE:]


def figure5_avalanches(kind: str, value: float, threshold: str) \
        -> tuple[np.ndarray, np.ndarray, int]:
    count = ({"gaussian": FIG5_GAUSSIAN_REPLICATES,
              "spikes": FIG5_SPIKE_REPLICATES}[kind][value])
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS / f"fig5_avalanches_{kind}_{number_label(value)}_{threshold}.npz"
    if path.exists():
        with np.load(path) as cached:
            return cached["duration"], cached["size"], int(cached["threshold"])
    histogram = np.zeros(201, dtype=np.int64)
    for replicate in range(count):
        activity = figure5_activity(kind, value, replicate)
        histogram += np.bincount(activity, minlength=201)
    theta = pooled_threshold(histogram, STABLE * count, threshold)
    durations, sizes = [], []
    for replicate in range(count):
        duration, size = reproduction.avalanches(
            figure5_activity(kind, value, replicate), theta
        )
        durations.append(duration); sizes.append(size)
    duration = np.concatenate(durations); size = np.concatenate(sizes)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, duration=duration, size=size,
                        threshold=np.array(theta))
    temporary.replace(path)
    return duration, size, theta


def analyze_and_render_figure5() -> None:
    import matplotlib.pyplot as plt
    from scipy.stats import binom

    distributions = {}
    summary = {}
    for kind, design in (("gaussian", FIG5_GAUSSIAN_REPLICATES),
                         ("spikes", FIG5_SPIKE_REPLICATES)):
        low_percentile = "5percent" if kind == "gaussian" else "10percent"
        for value, count in design.items():
            duration, size, theta = figure5_avalanches(kind, value, "half")
            unused_d, size_low, theta_low = figure5_avalanches(
                kind, value, low_percentile
            )
            unused_d, size_high, theta_high = figure5_avalanches(
                kind, value, "25percent"
            )
            histogram = np.zeros(201, dtype=np.int64)
            for replicate in range(count):
                histogram += np.bincount(
                    figure5_activity(kind, value, replicate), minlength=201
                )
            distributions[(kind, value)] = (size, size_low, size_high,
                                             histogram / histogram.sum())
            summary[f"{kind}_{value:g}"] = {
                "replicates": count, "half_mean_threshold": theta,
                "lower_percentile_threshold": theta_low,
                "percentile_25_threshold": theta_high,
                "avalanche_count": int(duration.size),
            }
    path = ANALYSIS / "Fig5_summary.json"
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(path)

    fig = plt.figure(figsize=(6, 9))
    grid = fig.add_gridspec(5, 2, height_ratios=(2, 2, 1, 1, 1))
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))
    schemes = {
        "gaussian": ((0.005, "darkcyan", "low"),
                     (0.05, "black", "intermediate"), (5.0, "red", "high")),
        "spikes": ((0.0, "darkcyan", "0%"),
                   (0.05, "black", "5%"), (0.10, "red", "10%")),
    }
    for row, kind in enumerate(("gaussian", "spikes")):
        size_ax, activity_ax = axes[row * 2], axes[row * 2 + 1]
        for value, color, label in schemes[kind]:
            size, size_low, size_high, activity_pdf = distributions[(kind, value)]
            x, y = log_binned_pdf(size)
            size_ax.loglog(x, y, color=color, lw=2 if color == "black" else 1.25)
            shade_log_pdf(size_ax, size_low, size_high, color)
            activity_ax.plot(np.arange(activity_pdf.size), activity_pdf,
                             color=color, lw=2 if color == "black" else 1.25,
                             label=label)
        size_ax.set_xlim(1, 3_000); size_ax.set_ylim(1e-5, 1e-1)
        size_ax.set_xlabel(r"$S$"); size_ax.set_ylabel(r"$f(S)$")
        activity_ax.set_xlim(0, 70); activity_ax.set_ylim(0, 0.11)
        activity_ax.set_xlabel("a(t) [# neurons]"); activity_ax.set_ylabel("p(a(t))")
        activity_ax.plot(np.arange(40), binom.pmf(np.arange(40), 200, 0.1),
                         "--", color="gray", lw=1.5)
        activity_ax.legend(frameon=False, title="Noise level", fontsize=8)
    for label, ax in zip("ABCD", axes):
        ax.spines[["right", "top"]].set_visible(False)
        ax.text(-0.22, 1.03, label, transform=ax.transAxes,
                fontweight="bold", fontsize=12)
    for row, (value, unused_color, label) in enumerate(schemes["gaussian"], start=2):
        ax = fig.add_subplot(grid[row, :])
        with np.load(fig5_cache_path("gaussian", value, 0)) as cached:
            spikes = cached["spikes"]
        neurons, times = np.nonzero(spikes)
        ax.scatter(times, neurons, s=0.2, color="black")
        ax.set_ylabel(label); ax.set_yticks([])
        ax.spines[["right", "top", "left"]].set_visible(False)
        if row == 4: ax.set_xlabel("Time step")
        else: ax.set_xticks([])
    fig.tight_layout()
    figures = PAPER_ROOT / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "Fig5.pdf"); fig.savefig(figures / "Fig5.png", dpi=300)
    plt.close(fig)
    print(f"rendered {figures / 'Fig5.pdf'}")


def concatenate_avalanches(activities: list[np.ndarray], threshold: int) \
        -> tuple[np.ndarray, np.ndarray]:
    durations, sizes = [], []
    for activity in activities:
        duration, size = reproduction.avalanches(activity, threshold)
        durations.append(duration); sizes.append(size)
    return np.concatenate(durations), np.concatenate(sizes)


def onset_avalanches(activities: list[np.ndarray], threshold: int,
                     onset_window: int = 20) -> tuple[np.ndarray, np.ndarray]:
    durations, sizes = [], []
    for activity in activities:
        above = activity > threshold
        edges = np.diff(np.r_[False, above, False].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for start, end in zip(starts, ends):
            if start >= onset_window:
                break
            durations.append(int(end - start))
            sizes.append(int(np.sum(activity[start:end] - threshold)))
    return np.asarray(durations), np.asarray(sizes)


def analyze_and_render_figure6() -> None:
    import matplotlib.pyplot as plt

    missing = [replicate for replicate in range(FIG6_REPLICATES)
               if not fig6_cache_path(replicate).exists()]
    if missing:
        raise SystemExit(f"Figure 6 is missing {len(missing)} replicates")
    normal, onset, readapted = [], [], []
    for replicate in range(FIG6_REPLICATES):
        with np.load(fig6_cache_path(replicate)) as cached:
            if replicate < 50:
                normal.append(cached["normal_activity"])
                readapted.append(cached["input_activity"][20:])
            onset.append(cached["input_activity"][:200])
    normal_values = np.concatenate(normal)
    threshold = int(normal_values.mean() / 2.0) + 1
    regimes = {
        "normal": concatenate_avalanches(normal, threshold),
        "onset": onset_avalanches(onset, threshold),
        "readaptation": concatenate_avalanches(readapted, threshold),
    }
    summary = {
        "normal_replicates": 50, "onset_replicates": FIG6_REPLICATES,
        "readaptation_replicates": 50, "section_steps": FIG6_SECTION_STEPS,
        "onset_window": 20, "onset_trace_saved": 200,
        "shared_activity_threshold": threshold,
        "avalanches": {key: int(value[0].size) for key, value in regimes.items()},
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary_path = ANALYSIS / "Fig6_summary.json"
    temporary = summary_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(summary_path)
    arrays = ANALYSIS / "Fig6_avalanches.npz"
    temporary_arrays = arrays.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_arrays, **{
        f"{regime}_{quantity}": values[index]
        for regime, values in regimes.items()
        for index, quantity in enumerate(("duration", "size"))
    })
    temporary_arrays.replace(arrays)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    styles = (("normal", "black", "Before input"),
              ("onset", "red", "Input onset"),
              ("readaptation", "c", "Readaptation"))
    for regime, color, label in styles:
        duration, size = regimes[regime]
        x, y = log_binned_pdf(duration); axes[0].loglog(x, y, color=color, lw=1.5)
        x, y = log_binned_pdf(size); axes[1].loglog(x, y, color=color, lw=1.5,
                                                       label=label)
    for ax, xlabel, ylabel, xmax, ymin in (
        (axes[0], r"$T$", r"$f(T)$", 300, 1e-4),
        (axes[1], r"$S$", r"$f(S)$", 3_000, 1e-5),
    ):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_xlim(1, xmax); ax.set_ylim(ymin, 1)
        ax.spines[["right", "top"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)
    for label, ax in zip("AB", axes):
        ax.text(-0.18, 1.02, label, transform=ax.transAxes,
                fontweight="bold", fontsize=12)
    fig.tight_layout()
    figures = PAPER_ROOT / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "Fig6.pdf"); fig.savefig(figures / "Fig6.png", dpi=300)
    plt.close(fig)
    print(f"rendered {figures / 'Fig6.pdf'}")


def readout_performance(states: np.ndarray, letters: np.ndarray, classes: int,
                        reduced: bool = False) -> float:
    split = FIG7_TRAIN_STEPS
    x_train = (states[:, :split] >= 0).astype(float)
    x_test = (states[:, split:split + FIG7_TEST_STEPS] >= 0).astype(float)
    y_train = letters[:split].astype(int)
    y_test = letters[split:split + FIG7_TEST_STEPS].astype(int)
    targets = np.eye(classes)[y_train].T
    weights = targets @ np.linalg.pinv(x_train)
    prediction = np.argmax(weights @ x_test, axis=0)
    if reduced:
        mask = np.isin(y_test, (1, 2, 4, 5))
        prediction = prediction[mask]; y_test = y_test[mask]
    return float(np.mean(prediction == y_test))


def summarize_performance(values: list[float], low: float,
                          high: float) -> dict[str, float]:
    data = np.asarray(values)
    return {"mean": float(data.mean()), "std": float(data.std()),
            "lower_percentile": float(np.percentile(data, low)),
            "upper_percentile": float(np.percentile(data, high))}


def analyze_and_render_figure7() -> None:
    import matplotlib.pyplot as plt

    missing_counting = [
        (variant, n, replicate) for variant in FIG7_COUNTING_VARIANTS
        for n in FIG7_COUNTING_N for replicate in range(FIG7_REPLICATES)
        if not fig7_counting_cache_path(variant, n, replicate).exists()
    ]
    missing_random = [
        (length, replicate) for length in FIG7_RANDOM_LENGTHS
        for replicate in range(FIG7_REPLICATES)
        if not fig7_random_cache_path(length, replicate).exists()
    ]
    if missing_counting or missing_random:
        raise SystemExit(f"Figure 7 missing counting={len(missing_counting)}, "
                         f"random={len(missing_random)}")
    performance_results = {"counting": {}, "random": {}}
    for variant in FIG7_COUNTING_VARIANTS:
        for n in FIG7_COUNTING_N:
            values = []
            for replicate in range(FIG7_REPLICATES):
                with np.load(fig7_counting_cache_path(variant, n, replicate)) as cached:
                    values.append(readout_performance(cached["states"], cached["letters"],
                                                      6, reduced=True))
            performance_results["counting"].setdefault(variant, {})[str(n)] = \
                summarize_performance(values, 16, 84)
    for length in FIG7_RANDOM_LENGTHS:
        values = []
        for replicate in range(FIG7_REPLICATES):
            with np.load(fig7_random_cache_path(length, replicate)) as cached:
                values.append(readout_performance(cached["states"], cached["letters"], 10))
        performance_results["random"][str(length)] = summarize_performance(values, 5, 95)

    avalanche_results = {}
    avalanche_data = {}
    for kind, values in (("counting", (4, 20)), ("random", (10, 20, 100))):
        for value in values:
            activities = []
            for replicate in range(FIG7_REPLICATES):
                path = (fig7_counting_cache_path("noise", value, replicate)
                        if kind == "counting" else
                        fig7_random_cache_path(value, replicate))
                with np.load(path) as cached:
                    activities.append(cached["avalanche_activity"])
            pooled = np.concatenate(activities)
            theta = int(pooled.mean() / 2.0) + 1
            duration, size = concatenate_avalanches(activities, theta)
            avalanche_data[(kind, value)] = duration, size
            avalanche_results[f"{kind}_{value}"] = {
                "threshold": theta, "avalanche_count": int(duration.size)
            }
    results = {"performance": performance_results,
               "avalanches": avalanche_results,
               "replicates": FIG7_REPLICATES}
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS / "Fig7_summary.json"
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(path)

    fig, axes = plt.subplots(4, 2, figsize=(7, 14))
    axes[1, 1].remove(); axes[3, 1].remove()
    for value, color in ((4, "blue"), (20, "green")):
        duration, size = avalanche_data[("counting", value)]
        x, y = log_binned_pdf(duration); axes[0, 0].loglog(x, y, color=color)
        x, y = log_binned_pdf(size); axes[0, 1].loglog(x, y, color=color,
                                                      label=rf"$n={value}$")
    count_ax = axes[1, 0]
    for variant, color, label in (("noise", "blue", r"$\sigma=0.05$"),
                                  ("no_noise", "red", r"$\sigma=0.00$"),
                                  ("original", "black", "Original SORN")):
        summaries = [performance_results["counting"][variant][str(n)]
                     for n in FIG7_COUNTING_N]
        means = np.asarray([item["mean"] for item in summaries])
        low = means - np.asarray([item["lower_percentile"] for item in summaries])
        high = np.asarray([item["upper_percentile"] for item in summaries]) - means
        count_ax.errorbar(FIG7_COUNTING_N, means, yerr=(low, high), color=color,
                          marker="o", label=label)
    for value, color in ((10, "blue"), (20, "green"), (100, "c")):
        duration, size = avalanche_data[("random", value)]
        x, y = log_binned_pdf(duration); axes[2, 0].loglog(x, y, color=color)
        x, y = log_binned_pdf(size); axes[2, 1].loglog(x, y, color=color,
                                                      label=rf"$L={value}$")
    random_ax = axes[3, 0]
    summaries = [performance_results["random"][str(length)]
                 for length in FIG7_RANDOM_LENGTHS]
    means = np.asarray([item["mean"] for item in summaries])
    low = means - np.asarray([item["lower_percentile"] for item in summaries])
    high = np.asarray([item["upper_percentile"] for item in summaries]) - means
    random_ax.errorbar(FIG7_RANDOM_LENGTHS, means, yerr=(low, high),
                       color="blue", marker="o")
    for ax in (axes[0, 0], axes[2, 0]):
        ax.set_xlim(1, 300); ax.set_ylim(1e-4, 1)
        ax.set_xlabel(r"$T$"); ax.set_ylabel(r"$f(T)$")
    for ax in (axes[0, 1], axes[2, 1]):
        ax.set_xlim(1, 3_000); ax.set_ylim(1e-5, 1e-1)
        ax.set_xlabel(r"$S$"); ax.set_ylabel(r"$f(S)$"); ax.legend(frameon=False)
    count_ax.set(xlabel="n", ylabel="Performance", xlim=(0, 25), ylim=(0.4, 1.1),
                 title="Counting Task"); count_ax.legend(frameon=False)
    random_ax.set(xlabel="L", ylabel="Performance", xlim=(0, 110), ylim=(0.4, 1.1),
                  title="Random Task")
    visible_axes = (axes[0, 0], axes[0, 1], count_ax,
                    axes[2, 0], axes[2, 1], random_ax)
    for label, ax in zip("ABCDEF", visible_axes):
        ax.spines[["right", "top"]].set_visible(False)
        ax.text(-0.2, 1.02, label, transform=ax.transAxes,
                fontweight="bold", fontsize=12)
    fig.tight_layout()
    figures = PAPER_ROOT / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "Fig7.pdf"); fig.savefig(figures / "Fig7.png", dpi=300)
    plt.close(fig)
    print(f"rendered {figures / 'Fig7.pdf'}")


def fit_figure2() -> Path:
    results: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replicates_per_network_size": REPLICATES,
        "transient_steps": TRANSIENT,
        "analysis_steps": STABLE,
        "threshold": "integer half of pooled mean activity",
        "network_sizes": {},
    }
    avalanche_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for network_size in NETWORK_SIZES:
        duration, size, threshold = load_avalanches(network_size)
        avalanche_data[network_size] = duration, size
        duration_range = FIT_RANGES[network_size]["duration"]
        size_range = FIT_RANGES[network_size]["size"]
        results["network_sizes"][str(network_size)] = {
            "avalanche_count": int(duration.size),
            "activity_threshold": threshold,
            "duration_power_law": power_law_result(duration, *duration_range),
            "size_power_law": power_law_result(size, *size_range),
        }

    duration, size = avalanche_data[200]
    duration_truncated, duration_fit = truncated_result(duration, 6)
    size_truncated, size_fit = truncated_result(size, 10)
    duration_power = results["network_sizes"]["200"]["duration_power_law"]
    size_power = results["network_sizes"]["200"]["size_power_law"]
    exponent_ratio = ((duration_power["alpha"] - 1.0) /
                      (size_power["alpha"] - 1.0))

    counts = np.bincount(duration)
    size_sums = np.bincount(duration, weights=size)
    unique_duration = np.flatnonzero(counts)
    mean_size = size_sums[unique_duration] / counts[unique_duration]
    mask = ((unique_duration >= FIT_RANGES[200]["duration"][0]) &
            (unique_duration <= FIT_RANGES[200]["duration"][1]) &
            (mean_size > 0))
    gamma_data, intercept = np.polyfit(
        np.log10(unique_duration[mask]), np.log10(mean_size[mask]), 1
    )
    results["n200_full_fits"] = {
        "duration_truncated_power_law": duration_truncated,
        "size_truncated_power_law": size_truncated,
        "duration_distribution_comparisons": {
            alternative: comparison(duration_fit, alternative)
            for alternative in ("exponential", "stretched_exponential")
        },
        "size_distribution_comparisons": {
            alternative: comparison(size_fit, alternative)
            for alternative in ("exponential", "stretched_exponential")
        },
        "theoretical_exponent_ratio": float(exponent_ratio),
        "gamma_data_loglog_slope": float(gamma_data),
        "gamma_data_log10_intercept": float(intercept),
        "gamma_data_fit_duration_range": list(FIT_RANGES[200]["duration"]),
    }
    reproduced = {
        "duration_power_law_alpha": duration_power["alpha"],
        "duration_truncated_alpha": duration_truncated["alpha"],
        "duration_truncated_beta": duration_truncated["beta"],
        "size_power_law_alpha": size_power["alpha"],
        "size_truncated_alpha": size_truncated["alpha"],
        "size_truncated_beta": size_truncated["beta"],
        "theoretical_exponent_ratio": float(exponent_ratio),
    }
    results["published_figure2_values"] = PUBLISHED_FIG2
    results["reproduction_minus_published"] = {
        key: float(reproduced[key] - PUBLISHED_FIG2[key])
        for key in reproduced
    }

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS / "Fig2_fits.json"
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(path)
    render_figure2(results, avalanche_data)
    print(f"wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--network-size", type=int, required=True)
    baseline.add_argument("--replicate", type=int, required=True)
    baseline.add_argument("--force", action="store_true")
    batch = subparsers.add_parser("run-figure2")
    batch.add_argument("--workers", type=int, default=7)
    figure4 = subparsers.add_parser("figure4")
    figure4.add_argument("--condition", choices=FIG4_CONDITIONS, required=True)
    figure4.add_argument("--replicate", type=int, required=True)
    figure4.add_argument("--force", action="store_true")
    figure4_batch = subparsers.add_parser("run-figure4")
    figure4_batch.add_argument("--workers", type=int, default=7)
    figure5 = subparsers.add_parser("figure5")
    figure5.add_argument("--kind", choices=("gaussian", "spikes"), required=True)
    figure5.add_argument("--value", type=float, required=True)
    figure5.add_argument("--replicate", type=int, required=True)
    figure5.add_argument("--force", action="store_true")
    figure5_batch = subparsers.add_parser("run-figure5")
    figure5_batch.add_argument("--workers", type=int, default=7)
    figure6 = subparsers.add_parser("figure6")
    figure6.add_argument("--replicate", type=int, required=True)
    figure6.add_argument("--force", action="store_true")
    figure6_batch = subparsers.add_parser("run-figure6")
    figure6_batch.add_argument("--workers", type=int, default=7)
    figure7_counting = subparsers.add_parser("figure7-counting")
    figure7_counting.add_argument("--variant", choices=FIG7_COUNTING_VARIANTS,
                                  required=True)
    figure7_counting.add_argument("--n", type=int, required=True)
    figure7_counting.add_argument("--replicate", type=int, required=True)
    figure7_counting.add_argument("--force", action="store_true")
    figure7_random = subparsers.add_parser("figure7-random")
    figure7_random.add_argument("--length", type=int, required=True)
    figure7_random.add_argument("--replicate", type=int, required=True)
    figure7_random.add_argument("--force", action="store_true")
    figure7_batch = subparsers.add_parser("run-figure7")
    figure7_batch.add_argument("--workers", type=int, default=7)
    remaining = subparsers.add_parser("run-figures5-7")
    remaining.add_argument("--workers", type=int, default=7)
    subparsers.add_parser("status")
    subparsers.add_parser("manifest")
    subparsers.add_parser("fit-figure2")
    subparsers.add_parser("render-figure2")
    subparsers.add_parser("render-figures1-3")
    subparsers.add_parser("analyze-figure4")
    subparsers.add_parser("analyze-figure5")
    subparsers.add_parser("analyze-figure6")
    subparsers.add_parser("analyze-figure7")
    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline(args.network_size, args.replicate, args.force)
    elif args.command == "run-figure2":
        run_figure2_batch(args.workers)
    elif args.command == "figure4":
        run_figure4_condition(args.condition, args.replicate, args.force)
    elif args.command == "run-figure4":
        run_figure4_batch(args.workers)
    elif args.command == "figure5":
        run_figure5_condition(args.kind, args.value, args.replicate, args.force)
    elif args.command == "run-figure5":
        run_figure5_batch(args.workers)
    elif args.command == "figure6":
        run_figure6_replicate(args.replicate, args.force)
    elif args.command == "run-figure6":
        run_figure6_batch(args.workers)
    elif args.command == "figure7-counting":
        run_figure7_counting(args.variant, args.n, args.replicate, args.force)
    elif args.command == "figure7-random":
        run_figure7_random(args.length, args.replicate, args.force)
    elif args.command == "run-figure7":
        run_figure7_batch(args.workers)
    elif args.command == "run-figures5-7":
        run_figures5_through7(args.workers)
    elif args.command == "status":
        print_status()
    elif args.command == "manifest":
        write_manifest()
    elif args.command == "fit-figure2":
        fit_figure2()
    elif args.command == "render-figure2":
        render_figure2_from_cache()
    elif args.command == "render-figures1-3":
        render_figures1_and3()
    elif args.command == "analyze-figure4":
        analyze_and_render_figure4()
    elif args.command == "analyze-figure5":
        analyze_and_render_figure5()
    elif args.command == "analyze-figure6":
        analyze_and_render_figure6()
    elif args.command == "analyze-figure7":
        analyze_and_render_figure7()


if __name__ == "__main__":
    main()
