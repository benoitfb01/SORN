import numpy as np

from differential.compare import compare_archives


def write_pair(tmp_path, left, right, key="case__R_x"):
    python2 = tmp_path / "python2.npz"
    python3 = tmp_path / "python3.npz"
    np.savez(python2, **{key: left})
    np.savez(python3, **{key: right})
    return python2, python3


def test_continuous_values_allow_only_roundoff(tmp_path):
    left = np.array([1.0, 2.0])
    python2, python3 = write_pair(tmp_path, left, left + 1e-14)
    assert compare_archives(python2, python3)["passed"]

    python2, python3 = write_pair(tmp_path, left, left + 1e-8)
    report = compare_archives(python2, python3)
    assert not report["passed"]
    assert report["first_divergence"]["array"] == "case__R_x"


def test_neuronal_states_must_match_exactly(tmp_path):
    left = np.array([[0, 1], [1, 0]])
    right = left.astype(float)
    right[1, 0] += 1e-14
    python2, python3 = write_pair(tmp_path, left, right, key="case__x")
    assert not compare_archives(python2, python3)["passed"]
