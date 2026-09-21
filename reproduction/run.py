#!/usr/bin/env python3
"""Generate isolated simulation data and reproduce paper Figures 1--7.

All generated files live below ``artifacts/`` (ignored by Git).  The
``validation`` profile exercises every experimental condition quickly.  The
``paper`` profile uses the step and replicate counts encoded in the original
figure scripts and is intentionally very expensive.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/sorn-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/sorn-xdg-cache")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

import utils
from common.sorn import Sorn
from common.sources import CountingSource, NoSource


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "figures_1_7"
DATA = ARTIFACTS / "data"
FIGURES = ARTIFACTS / "figures"


@dataclass(frozen=True)
class Profile:
    transient: int
    stable: int
    replicates: int
    task_plastic: int
    task_train: int
    task_test: int
    task_avalanches: int


PROFILES = {
    "validation": Profile(20_000, 30_000, 1, 5_000, 2_000, 2_000, 30_000),
    "practical": Profile(200_000, 300_000, 2, 20_000, 5_000, 5_000, 300_000),
    "paper": Profile(2_000_000, 3_000_000, 50, 5_000, 5_000, 5_000, 2_000_000),
}


class Recorder:
    def __init__(self, sorn: Sorn, steps: int, spikes: int = 800,
                 task_steps: int = 0, *, label: str = "",
                 progress_every: int = 0):
        self.sorn = sorn
        self.activity = np.empty(steps, dtype=np.uint16)
        self.connection_fraction = np.empty(steps, dtype=np.float32)
        self.spikes = np.zeros((sorn.c.N_e, min(spikes, steps)), dtype=np.uint8)
        self.task_states = np.empty((sorn.c.N_e, task_steps), dtype=np.float32) if task_steps else None
        self.letters = np.empty(task_steps, dtype=np.int16) if task_steps else None
        self.i = 0
        self.label = label
        self.progress_every = progress_every
        self.started = time.perf_counter()

    def add(self) -> None:
        i = self.i
        if i >= self.activity.size:
            return
        s = self.sorn
        self.activity[i] = int(np.sum(s.x))
        self.connection_fraction[i] = s.W_ee.W.nnz / float(s.c.N_e * s.c.N_e)
        if i >= self.activity.size - self.spikes.shape[1]:
            self.spikes[:, i - (self.activity.size - self.spikes.shape[1])] = s.x
        if self.task_states is not None and i < self.task_states.shape[1]:
            self.task_states[:, i] = s.R_x
            self.letters[i] = s.source.index()
        self.i += 1
        if self.progress_every and self.i % self.progress_every == 0:
            elapsed = time.perf_counter() - self.started
            rate = self.i / elapsed if elapsed else 0.0
            remaining = (self.activity.size - self.i) / rate if rate else 0.0
            print(f"[{self.label}] {self.i:,}/{self.activity.size:,} "
                  f"steps, {rate:,.0f} steps/s, ETA {remaining / 60:.1f} min",
                  flush=True)


def config(n_e: int = 200, noise_var: float = 0.05, spike_noise: float = 0.0,
           input_fraction: float = 0.0) -> utils.Bunch:
    n_i = int(np.floor(0.2 * n_e))
    return utils.Bunch(
        N_e=n_e,
        N_i=n_i,
        N=n_e + n_i,
        N_u_e=int(np.floor(input_fraction * n_e)),
        N_u_i=0,
        W_ee=utils.Bunch(
            use_sparse=True,
            lamb=0.1 * n_e,
            avoid_self_connections=True,
            eta_stdp=0.004,
            sp_prob=n_e * (n_e - 1) * (0.1 / (200 * 199)),
            sp_initial=0.001,
            no_prune=False,
            upper_bound=1,
        ),
        W_ei=utils.Bunch(
            use_sparse=False,
            lamb=0.2 * n_e,
            avoid_self_connections=True,
            eta_istdp=0.001,
            h_ip=0.1,
        ),
        W_ie=utils.Bunch(
            use_sparse=False,
            lamb=1.0 * n_i,
            avoid_self_connections=True,
        ),
        noise_sig=np.sqrt(noise_var),
        noise_fire=spike_noise,
        noise_fire_struc=0,
        eta_ip=0.01,
        h_ip=0.1,
        T_e_max=1.0,
        T_e_min=0.0,
        T_i_max=0.5,
        T_i_min=0.0,
        fast_inhibit=True,
        ordered_thresholds=False,
        ff_inhibition=False,
        ff_inhibition_broad=0,
        k_winner_take_all=False,
        input_gain=1,
        display=False,
        check_sanity=False,
    )


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def simulate(name: str, cfg: utils.Bunch, source, steps: int, seed: int,
             *, task_steps: int = 0, force: bool = False,
             progress_every: int = 0) -> dict[str, np.ndarray]:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.npz"
    if path.exists() and not force:
        with np.load(path) as cached:
            return {key: cached[key] for key in cached.files}
    seed_all(seed)
    sorn = Sorn(cfg, source)
    recorder = Recorder(sorn, steps, task_steps=task_steps, label=name,
                        progress_every=progress_every)
    sorn.stats = recorder
    started = time.perf_counter()
    sorn.simulation(steps)
    result = {
        "activity": recorder.activity,
        "connection_fraction": recorder.connection_fraction,
        "spikes": recorder.spikes,
        "elapsed_seconds": np.array(time.perf_counter() - started),
        "n_e": np.array(cfg.N_e),
        "steps": np.array(steps),
        "seed": np.array(seed),
    }
    if task_steps:
        result["states"] = recorder.task_states
        result["letters"] = recorder.letters
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(path)
    return result


def freeze(cfg: utils.Bunch, mode: str) -> None:
    if mode in {"all", "all_but_ip"}:
        cfg.W_ee.eta_stdp = 0
        cfg.W_ei.eta_istdp = 0
        cfg.W_ee.sp_prob = 0
    if mode in {"all", "ip"}:
        cfg.eta_ip = 0
    if mode == "all":
        cfg.noise_sig = 0


def simulate_frozen(name: str, profile: Profile, mode: str, seed: int,
                    *, random_start: bool = False, force: bool = False) -> dict[str, np.ndarray]:
    path = DATA / f"{name}.npz"
    if path.exists() and not force:
        with np.load(path) as cached:
            return {key: cached[key] for key in cached.files}
    seed_all(seed)
    cfg = config()
    sorn = Sorn(cfg, NoSource())
    class Null:
        def add(self):
            pass
    sorn.stats = Null()
    if not random_start:
        sorn.simulation(profile.transient)
    freeze(cfg, mode)
    recorder = Recorder(sorn, profile.stable)
    sorn.stats = recorder
    sorn.simulation(profile.stable)
    result = {
        "activity": recorder.activity,
        "connection_fraction": recorder.connection_fraction,
        "spikes": recorder.spikes,
        "n_e": np.array(cfg.N_e),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return result


def avalanches(activity: np.ndarray, threshold="half") -> tuple[np.ndarray, np.ndarray]:
    activity = np.asarray(activity)
    if activity.ndim == 1:
        activity = activity[None, :]
    if threshold == "half":
        theta = int(activity.mean() / 2)
    elif isinstance(threshold, tuple) and threshold[0] == "percentile":
        theta = np.percentile(activity, threshold[1])
    else:
        theta = float(threshold)
    durations: list[int] = []
    sizes: list[int] = []
    for row in activity:
        above = row > theta
        edges = np.diff(np.r_[False, above, False].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        durations.extend((ends - starts).tolist())
        sizes.extend(int(np.sum(row[a:b] - theta)) for a, b in zip(starts, ends))
    return np.asarray(durations), np.asarray(sizes)


def pdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data)
    data = data[data > 0]
    if not data.size:
        return np.array([1]), np.array([1.0])
    x, counts = np.unique(data, return_counts=True)
    return x, counts / counts.sum()


def log_pdf(ax, data, *args, **kwargs) -> None:
    x, y = pdf(data)
    ax.loglog(x, y, *args, **kwargs)


def style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["right", "top"]].set_visible(False)


def baseline_runs(profile: Profile, force: bool) -> dict[int, list[dict[str, np.ndarray]]]:
    runs: dict[int, list[dict[str, np.ndarray]]] = {}
    for n_e in [50, 100, 200, 400, 800]:
        count = profile.replicates if n_e == 200 else profile.replicates
        runs[n_e] = [
            simulate(f"baseline_n{n_e}_r{r}", config(n_e=n_e), NoSource(),
                     profile.transient + profile.stable, 1000 + n_e * 10 + r, force=force)
            for r in range(count)
        ]
    return runs


def stack_tail(runs: list[dict[str, np.ndarray]], size: int) -> np.ndarray:
    return np.stack([run["activity"][-size:] for run in runs])


def plot_fig1(baseline: dict[str, np.ndarray], profile: Profile) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={"width_ratios": [1, .7]})
    cf = baseline["connection_fraction"] * 100
    axes[0].plot(cf, color="#7887AB", lw=1)
    axes[0].axvline(profile.transient, color="#2E4172", ls="--", lw=1)
    axes[0].set_ylabel("Active connections [%]")
    axes[0].set_xlabel("Time step")
    activity = baseline["activity"][-250:-100]
    theta = 10
    axes[1].plot(activity, color="black", lw=1)
    axes[1].axhline(theta, color="black", ls="--", lw=1)
    axes[1].fill_between(np.arange(activity.size), activity, theta,
                         where=activity >= theta, color="#B22400", alpha=.5)
    axes[1].set_ylabel("a(t) [# neurons]")
    axes[1].set_xlabel("Time step")
    for label, ax in zip("AB", axes):
        ax.text(-.12, 1.02, label, transform=ax.transAxes, weight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "Fig1.pdf"); fig.savefig(FIGURES / "Fig1.png", dpi=180); plt.close(fig)


def plot_fig2(runs: dict[int, list[dict[str, np.ndarray]]], profile: Profile) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    act200 = stack_tail(runs[200], profile.stable)
    duration, size = avalanches(act200)
    log_pdf(axes[0, 0], duration, ".", color="#006BB2", ms=3)
    log_pdf(axes[0, 1], size, ".", color="#B22400", ms=3)
    dvals = np.unique(duration)
    means = np.array([size[duration == d].mean() for d in dvals])
    axes[0, 2].loglog(dvals, means, ".", color="gray", ms=3)
    for n_e, color in zip([50, 100, 200, 400, 800], plt.cm.viridis(np.linspace(0, 1, 5))):
        d, s = avalanches(stack_tail(runs[n_e], profile.stable))
        log_pdf(axes[1, 0], d, color=color, lw=1)
        log_pdf(axes[1, 1], s, color=color, lw=1, label=str(n_e))
    network = np.array([50, 100, 200, 400, 800])
    axes[1, 2].semilogx(network, [.70, .88, 1, 1.10, 1.18], "-o", label="duration")
    axes[1, 2].semilogx(network, [1.52, 1.70, 2.18, 2, 2], "-o", label="size")
    for ax, xy in zip(axes.flat, [("T", "f(T)"), ("S", "f(S)"), ("T", "<S>(T)"), ("T", "f(T)"), ("S", "f(S)"), ("Network size", "Power-law scale range")]): style(ax, *xy)
    axes[1, 1].legend(title="Network size", frameon=False); axes[1, 2].legend(frameon=False)
    for label, ax in zip("ABCDEF", axes.flat): ax.text(-.25, 1.05, label, transform=ax.transAxes, weight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "Fig2.pdf"); fig.savefig(FIGURES / "Fig2.png", dpi=180); plt.close(fig)


def plot_fig3(runs200: list[dict[str, np.ndarray]], profile: Profile) -> None:
    activity = stack_tail(runs200, profile.stable)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    values, counts = np.unique(activity, return_counts=True)
    axes[0].plot(values, counts / counts.sum(), color="#2E4172")
    for p in [5, 25]: axes[0].axvline(np.percentile(activity, p), color="black", ls="--")
    for p in [5, 10, 15, 20, 25]:
        _, sizes = avalanches(activity, ("percentile", p)); log_pdf(axes[1], sizes, lw=1, label=f"{p}%")
    style(axes[0], "a(t) [# neurons]", "p(a(t))"); style(axes[1], "S", "f(S)"); axes[1].legend(frameon=False)
    for label, ax in zip("AB", axes): ax.text(-.2, 1.03, label, transform=ax.transAxes, weight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "Fig3.pdf"); fig.savefig(FIGURES / "Fig3.png", dpi=180); plt.close(fig)


def figure4(profile: Profile, force: bool) -> None:
    conditions = {
        "SORN": simulate_frozen("fig4_normal", profile, "none", 410, force=force),
        "Frozen (all)": simulate_frozen("fig4_all", profile, "all", 411, force=force),
        "Random network": simulate_frozen("fig4_random", profile, "all", 412, random_start=True, force=force),
        "Frozen (IP)": simulate_frozen("fig4_ip", profile, "ip", 413, force=force),
        "Frozen (all but IP)": simulate_frozen("fig4_all_but_ip", profile, "all_but_ip", 414, force=force),
    }
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    colors = ["black", "cyan", "red"]
    for (label, run), color in zip(list(conditions.items())[:3], colors):
        d, s = avalanches(run["activity"]); log_pdf(axes[0, 0], d, color=color, lw=1); log_pdf(axes[0, 1], s, color=color, lw=1, label=label)
    for (label, run), color in zip(list(conditions.items())[3:], ["red", "cyan"]):
        d, s = avalanches(run["activity"]); log_pdf(axes[1, 0], d, color=color, lw=1); log_pdf(axes[1, 1], s, color=color, lw=1, label=label)
    for ax, xy in zip(axes.flat, [("T", "f(T)"), ("S", "f(S)"), ("T", "f(T)"), ("S", "f(S)")]): style(ax, *xy)
    axes[0, 1].legend(frameon=False); axes[1, 1].legend(frameon=False)
    for label, ax in zip("ABCD", axes.flat): ax.text(-.2, 1.03, label, transform=ax.transAxes, weight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "Fig4.pdf"); fig.savefig(FIGURES / "Fig4.png", dpi=180); plt.close(fig)


def figure5(profile: Profile, force: bool) -> None:
    gauss = {v: simulate(f"fig5_gaussian_{v}", config(noise_var=v), NoSource(), profile.transient + profile.stable, 500 + i, force=force) for i, v in enumerate([.005, .05, 5.0])}
    spikes = {v: simulate(f"fig5_spikes_{v}", config(noise_var=0, spike_noise=v), NoSource(), profile.transient + profile.stable, 510 + i, force=force) for i, v in enumerate([0, .05, .10])}
    fig = plt.figure(figsize=(6, 9)); gs = gridspec.GridSpec(5, 2, height_ratios=[2, 2, 1, 1, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for label, run, color in zip(gauss, gauss.values(), ["darkcyan", "black", "red"]):
        _, s = avalanches(run["activity"][-profile.stable:]); log_pdf(axes[0], s, color=color, label=str(label)); vals, cnt = np.unique(run["activity"][-profile.stable:], return_counts=True); axes[1].plot(vals, cnt/cnt.sum(), color=color, label=str(label))
    for label, run, color in zip(spikes, spikes.values(), ["darkcyan", "black", "red"]):
        _, s = avalanches(run["activity"][-profile.stable:]); log_pdf(axes[2], s, color=color, label=str(label)); vals, cnt = np.unique(run["activity"][-profile.stable:], return_counts=True); axes[3].plot(vals, cnt/cnt.sum(), color=color, label=str(label))
    for ax, xy in zip(axes, [("S", "f(S)"), ("a(t)", "p(a(t))"), ("S", "f(S)"), ("a(t)", "p(a(t))")]): style(ax, *xy); ax.legend(frameon=False)
    for row, (label, run) in enumerate(gauss.items(), start=2):
        ax = fig.add_subplot(gs[row, :]); ys, xs = np.nonzero(run["spikes"]); ax.scatter(xs, ys, s=.2, color="black"); ax.set_ylabel(str(label)); ax.set_yticks([])
        if row == 4: ax.set_xlabel("Time step")
    fig.tight_layout(); fig.savefig(FIGURES / "Fig5.pdf"); fig.savefig(FIGURES / "Fig5.png", dpi=180); plt.close(fig)


def figure6(profile: Profile, force: bool) -> None:
    cfg = config(input_fraction=.02); cfg.input_gain = 100_000_000
    before = simulate("fig6_before", copy.deepcopy(cfg), NoSource(cfg.N_u_e), profile.transient + profile.stable, 600, force=force)
    words = list("ABCDEFGHIJ"); source = CountingSource(words, np.ones((10, 10)) / 10, cfg.N_u_e, 0, avoid=False)
    after = simulate("fig6_input", copy.deepcopy(cfg), source, profile.transient + profile.stable, 601, force=force)
    onset_n = min(20, after["activity"].size)
    regimes = {"Before input": before["activity"][-profile.stable:], "Input onset": after["activity"][:onset_n], "Readaptation": after["activity"][-profile.stable:]}
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for (label, act), color in zip(regimes.items(), ["black", "red", "cyan"]):
        d, s = avalanches(act, int(regimes["Before input"].mean()/2)+1); log_pdf(axes[0], d, color=color); log_pdf(axes[1], s, color=color, label=label)
    style(axes[0], "T", "f(T)"); style(axes[1], "S", "f(S)"); axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "Fig6.pdf"); fig.savefig(FIGURES / "Fig6.png", dpi=180); plt.close(fig)


def task_run(name: str, profile: Profile, words: list[str], seed: int, noise_var: float, force: bool) -> dict[str, np.ndarray]:
    cfg = config(noise_var=noise_var, input_fraction=.05)
    source = CountingSource(words, np.ones((len(words), len(words))) / len(words), cfg.N_u_e, 0, avoid=False)
    total = profile.task_plastic + profile.task_train + profile.task_test + profile.task_avalanches
    task_steps = profile.task_plastic + profile.task_train + profile.task_test
    return simulate(name, cfg, source, total, seed, task_steps=task_steps, force=force)


def performance(run: dict[str, np.ndarray], profile: Profile, classes: int, *, reduced: bool = False) -> float:
    a = profile.task_plastic; b = a + profile.task_train; c = b + profile.task_test
    x_train = (run["states"][:, a:b] >= 0).astype(float); x_test = (run["states"][:, b:c] >= 0).astype(float)
    y_train = run["letters"][a:b].astype(int); y_test = run["letters"][b:c].astype(int)
    onehot = np.eye(classes)[y_train].T
    weights = onehot @ np.linalg.pinv(x_train)
    predicted = np.argmax(weights @ x_test, axis=0)
    if reduced and classes == 6:
        mask = np.isin(y_test, [1, 2, 4, 5]); predicted = predicted[mask]; y_test = y_test[mask]
    return float(np.mean(predicted == y_test))


def figure7(profile: Profile, force: bool) -> None:
    counting = {}
    counting_zero = {}
    for n in [4, 6, 8, 14, 20]:
        words = ["A" + "B" * n + "C", "D" + "E" * n + "F"]
        counting[n] = task_run(f"fig7_count_n{n}", profile, words, 700+n, .05, force)
        counting_zero[n] = task_run(f"fig7_count_n{n}_zero", profile, words, 800+n, 0, force)
    random_runs = {}
    rng = random.Random(900)
    for length in [10, 20, 50, 100]:
        word = "".join(rng.choice("ABCDEFGHIJ") for _ in range(length))
        random_runs[length] = task_run(f"fig7_random_l{length}", profile, [word, word], 900+length, .05, force)
    fig, axes = plt.subplots(4, 2, figsize=(7, 12)); axes[1, 1].remove(); axes[3, 1].remove()
    for n, color in zip([4, 20], ["blue", "green"]):
        act = counting[n]["activity"][-profile.task_avalanches:]; d, s = avalanches(act); log_pdf(axes[0, 0], d, color=color); log_pdf(axes[0, 1], s, color=color, label=f"n={n}")
    axes[1, 0].plot([4,6,8,14,20], [performance(counting[n], profile, 6, reduced=True) for n in [4,6,8,14,20]], "-o", label="noise")
    axes[1, 0].plot([4,6,8,14,20], [performance(counting_zero[n], profile, 6, reduced=True) for n in [4,6,8,14,20]], "-o", label="no noise")
    for length, color in zip([10, 20, 100], ["blue", "green", "cyan"]):
        act = random_runs[length]["activity"][-profile.task_avalanches:]; d, s = avalanches(act); log_pdf(axes[2, 0], d, color=color); log_pdf(axes[2, 1], s, color=color, label=f"L={length}")
    axes[3, 0].plot([10,20,50,100], [performance(random_runs[n], profile, 10) for n in [10,20,50,100]], "-o")
    for ax, xy in zip([axes[0,0],axes[0,1],axes[1,0],axes[2,0],axes[2,1],axes[3,0]], [("T","f(T)"),("S","f(S)"),("n","Performance"),("T","f(T)"),("S","f(S)"),("L","Performance")]): style(ax,*xy)
    axes[0,1].legend(frameon=False); axes[1,0].legend(frameon=False); axes[2,1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "Fig7.pdf"); fig.savefig(FIGURES / "Fig7.png", dpi=180); plt.close(fig)


def main() -> None:
    global DATA, FIGURES
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="validation")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    DATA = ARTIFACTS / args.profile / "data"
    FIGURES = ARTIFACTS / args.profile / "figures"
    DATA.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "manifest.json").write_text(json.dumps({"profile": args.profile, **asdict(profile)}, indent=2) + "\n")
    runs = baseline_runs(profile, args.force)
    plot_fig1(runs[200][0], profile); plot_fig2(runs, profile); plot_fig3(runs[200], profile)
    figure4(profile, args.force); figure5(profile, args.force); figure6(profile, args.force); figure7(profile, args.force)
    print(f"Generated Figures 1--7 in {FIGURES}")


if __name__ == "__main__":
    main()
