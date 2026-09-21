#!/usr/bin/env python
"""Run controlled SORN core cases under either Python 2 or Python 3.

This file intentionally uses syntax supported by Python 2.7.  Stochastic
inputs are supplied by deterministic providers so library RNG changes cannot
hide or create algorithmic differences.
"""

from __future__ import division, print_function

import os
import sys
import types

import numpy as np


def install_python2_shims(source_root):
    """Provide only the non-core legacy imports needed by the untouched code."""
    pylab = types.ModuleType("pylab")
    for name in dir(np):
        setattr(pylab, name, getattr(np, name))
    for name in ("rand", "randn", "randint", "shuffle", "seed"):
        setattr(pylab, name, getattr(np.random, name))
    # Original sorn.py mixes pylab's unqualified names with ``np.*`` without
    # importing NumPy itself.  Historical pylab exposed NumPy as ``np``.
    pylab.np = np
    pylab.sys = sys
    # Copying NumPy's attributes also copies its restrictive ``__all__``.
    # The legacy modules expect pylab's import-star surface, including the
    # random helpers above, so define that surface explicitly.
    pylab.__all__ = [name for name in pylab.__dict__
                     if not name.startswith("_")]
    sys.modules["pylab"] = pylab

    class Bunch(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

    class DataLog(object):
        pass

    utils = types.ModuleType("utils")
    utils.Bunch = Bunch
    utils.DataLog = DataLog
    utils.backup = lambda *args, **kwargs: None
    sys.modules["utils"] = utils
    sys.path.insert(0, os.path.join(source_root, "common"))
    import sorn
    import sources
    import synapses
    return Bunch, sorn.Sorn, sources.NoSource, synapses


def import_core():
    if os.environ.get("SORN_RUNTIME") == "python2":
        return install_python2_shims(os.environ["SORN_SOURCE_ROOT"])
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    import utils
    from common.sorn import Sorn
    from common.sources import NoSource
    from common import synapses
    return utils.Bunch, Sorn, NoSource, synapses


Bunch, Sorn, NoSource, synapses = import_core()


class FixedSource(object):
    def __init__(self, sequence, w_eu, eta_stdp):
        self.sequence = np.asarray(sequence, dtype=float)
        self.w_eu = np.asarray(w_eu, dtype=float)
        self.eta_stdp = eta_stdp
        self.position = 0

    def next(self):
        value = self.sequence[self.position % len(self.sequence)].copy()
        self.position += 1
        return value

    __next__ = next

    def generate_connection_e(self, n_e):
        cfg = Bunch(use_sparse=False, lamb=np.inf,
                    avoid_self_connections=False,
                    eta_stdp=self.eta_stdp)
        matrix = synapses.create_matrix((n_e, self.w_eu.shape[1]), cfg)
        matrix.set_synapses(self.w_eu)
        return matrix

    def generate_connection_i(self, n_i):
        cfg = Bunch(use_sparse=False, lamb=np.inf,
                    avoid_self_connections=False)
        matrix = synapses.create_matrix((n_i, self.w_eu.shape[1]), cfg)
        matrix.set_synapses(np.zeros((n_i, self.w_eu.shape[1])))
        return matrix

    def global_range(self):
        return self.w_eu.shape[1]

    def global_index(self):
        return self.position % self.w_eu.shape[1]


class NullStats(object):
    def add(self):
        pass


class NoiseProvider(object):
    def __init__(self, n_e, n_i):
        self.n_e = n_e
        self.n_i = n_i
        self.calls = 0

    def randn(self, size):
        step = self.calls // 2
        self.calls += 1
        indices = np.arange(size, dtype=float)
        phase = 0.37 if size == self.n_e else 0.91
        return np.sin(indices * 0.73 + step * 0.41 + phase)


class StructuralProvider(object):
    def __init__(self, pairs):
        self.values = [item for pair in pairs for item in pair]
        self.position = 0

    def rand(self, *shape):
        return 0.0

    def randint(self, maximum):
        value = self.values[self.position]
        self.position += 1
        return value % maximum


def make_config(noise, plastic):
    n_e, n_i = 6, 2
    eta_stdp = 0.004 if plastic else 0.0
    eta_istdp = 0.001 if plastic else 0.0
    eta_ip = 0.01 if plastic else 0.0
    return Bunch(
        N_e=n_e, N_i=n_i, N=n_e + n_i,
        W_ee=Bunch(use_sparse=True, lamb=2,
                   avoid_self_connections=True, eta_stdp=eta_stdp,
                   sp_prob=0.0, sp_initial=0.001,
                   no_prune=False, upper_bound=1),
        W_ei=Bunch(use_sparse=False, lamb=2,
                   avoid_self_connections=True, eta_istdp=eta_istdp,
                   h_ip=0.25),
        W_ie=Bunch(use_sparse=False, lamb=2,
                   avoid_self_connections=True),
        noise_sig=noise, noise_fire=0, noise_fire_struc=0,
        eta_ip=eta_ip, h_ip=0.25,
        T_e_max=1.0, T_e_min=0.0,
        T_i_max=0.5, T_i_min=0.0,
        fast_inhibit=True, ordered_thresholds=False,
        ff_inhibition=False, ff_inhibition_broad=0,
        k_winner_take_all=False, input_gain=0.35,
        display=False, check_sanity=True,
    )


def fixtures():
    n_e = 6
    w_ee = np.zeros((n_e, n_e))
    for row in range(n_e):
        w_ee[row, (row - 1) % n_e] = 0.4
        w_ee[row, (row + 1) % n_e] = 0.6
    w_ei = np.asarray([[0.35, 0.65], [0.55, 0.45], [0.25, 0.75],
                       [0.70, 0.30], [0.40, 0.60], [0.60, 0.40]])
    w_ie = np.asarray([[.10, .20, .10, .25, .15, .20],
                       [.20, .10, .25, .10, .20, .15]])
    w_eu = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1], [.5, .5], [.5, .5]])
    sequence = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1], [0, 0]], dtype=float)
    return w_ee, w_ei, w_ie, w_eu, sequence


def dense(matrix):
    return np.asarray(matrix.get_synapses(), dtype=float)


def capture(sorn):
    return [sorn.x.copy(), sorn.y.copy(), sorn.R_x.copy(), sorn.R_y.copy(),
            sorn.T_e.copy(), sorn.T_i.copy(), dense(sorn.W_ee),
            dense(sorn.W_ei), dense(sorn.W_ie), dense(sorn.W_eu)]


def run_scenario(name, noise, plastic, steps):
    cfg = make_config(noise, plastic)
    w_ee, w_ei, w_ie, w_eu, sequence = fixtures()
    source = FixedSource(sequence, w_eu, 0.003 if plastic else 0.0)
    np.random.seed(123)
    sorn = Sorn(cfg, source)
    sorn.stats = NullStats()
    sorn.W_ee.set_synapses(w_ee)
    sorn.W_ei.set_synapses(w_ei)
    sorn.W_ie.set_synapses(w_ie)
    sorn.W_eu.set_synapses(w_eu)
    sorn.x = np.asarray([1, 0, 1, 0, 0, 1], dtype=int)
    sorn.y = np.asarray([0, 1], dtype=int)
    sorn.u = sequence[0].copy()
    source.position = 1
    sorn.R_x = np.zeros(cfg.N_e)
    sorn.R_y = np.zeros(cfg.N_i)
    sorn.T_e = np.asarray([.20, .30, .40, .50, .60, .70])
    sorn.T_i = np.asarray([.20, .35])

    original_randn = np.random.randn
    provider = NoiseProvider(cfg.N_e, cfg.N_i)
    np.random.randn = provider.randn
    snapshots = [capture(sorn)]
    try:
        for unused in range(steps):
            sorn.step(source.next())
            snapshots.append(capture(sorn))
    finally:
        np.random.randn = original_randn

    names = ["x", "y", "R_x", "R_y", "T_e", "T_i",
             "W_ee", "W_ei", "W_ie", "W_eu"]
    return dict((name + "__" + key,
                 np.asarray([snapshot[i] for snapshot in snapshots]))
                for i, key in enumerate(names))


def structural_case():
    cfg = make_config(0.0, True)
    w_ee, unused_ei, unused_ie, unused_eu, unused_seq = fixtures()
    cfg.W_ee.sp_prob = 1.0
    np.random.seed(321)
    matrix = synapses.create_matrix((cfg.N_e, cfg.N_e), cfg.W_ee)
    matrix.set_synapses(w_ee)
    pairs = [(i % 6, (i + 2) % 6) for i in range(11)]
    provider = StructuralProvider(pairs)
    old_rand, old_randint = np.random.rand, np.random.randint
    np.random.rand, np.random.randint = provider.rand, provider.randint
    try:
        for unused in range(11):
            matrix.struct_p()
        matrix.ss()
    finally:
        np.random.rand, np.random.randint = old_rand, old_randint
    return {"structural__W_ee": dense(matrix)}


def assert_sensitivity(output):
    """Fail if a scenario stops exercising the behavior it claims to test."""
    for case in ("plastic_deterministic", "plastic_noise_input"):
        for quantity in ("x", "y", "T_e", "W_ee", "W_ei", "W_eu"):
            trajectory = output[case + "__" + quantity]
            if np.array_equal(trajectory[0], trajectory[-1]):
                raise AssertionError(case + " did not change " + quantity)

    for quantity in ("T_e", "W_ee", "W_ei", "W_eu"):
        trajectory = output["frozen_noise_input__" + quantity]
        if not np.array_equal(trajectory[0], trajectory[-1]):
            raise AssertionError("frozen scenario changed " + quantity)

    initial_w_ee = fixtures()[0]
    if np.count_nonzero(output["structural__W_ee"] - initial_w_ee) < 6:
        raise AssertionError("structural plasticity did not insert connections")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: core_case.py OUTPUT.npz")
    output = {}
    output.update(run_scenario("plastic_deterministic", 0.0, True, 30))
    output.update(run_scenario("plastic_noise_input", 0.125, True, 30))
    output.update(run_scenario("frozen_noise_input", 0.125, False, 30))
    output.update(structural_case())
    assert_sensitivity(output)
    np.savez_compressed(sys.argv[1], **output)
    print("wrote", sys.argv[1], "arrays", len(output))


if __name__ == "__main__":
    main()
