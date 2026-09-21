import numpy as np

from reproduction.run import avalanches, config
from reproduction.paper_batch import FIT_RANGES, NETWORK_SIZES, seed_for


def test_avalanche_duration_and_size():
    activity = np.array([0, 2, 3, 0, 4, 4, 4, 0])
    duration, size = avalanches(activity, threshold=1)
    np.testing.assert_array_equal(duration, [2, 3])
    np.testing.assert_array_equal(size, [3, 9])


def test_network_size_configuration_scales_populations():
    cfg = config(n_e=400)
    assert cfg.N_e == 400
    assert cfg.N_i == 80
    assert cfg.N == 480
    assert cfg.W_ee.lamb == 40


def test_figure2_publication_design_and_seeds():
    assert NETWORK_SIZES == (50, 100, 200, 400, 800)
    assert FIT_RANGES[200]["duration"] == (6, 60)
    assert FIT_RANGES[200]["size"] == (10, 1500)
    seeds = {seed_for(network_size, replicate)
             for network_size in NETWORK_SIZES for replicate in range(50)}
    assert len(seeds) == 250
