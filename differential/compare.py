#!/usr/bin/env python3
"""Compare Python 2 reference and Python 3 port core trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DISCRETE = {"x", "y"}
RTOL = 1e-12
ATOL = 1e-12


def compare_archives(python2: str | Path, python3: str | Path) -> dict:
    report = {
        "passed": True,
        "comparison": {"discrete": "exact", "rtol": RTOL, "atol": ATOL},
        "arrays": {},
        "first_divergence": None,
    }
    with np.load(python2) as py2, np.load(python3) as py3:
        if set(py2.files) != set(py3.files):
            missing_from_python3 = sorted(set(py2.files) - set(py3.files))
            missing_from_python2 = sorted(set(py3.files) - set(py2.files))
            report["passed"] = False
            report["key_mismatch"] = {
                "missing_from_python3": missing_from_python3,
                "missing_from_python2": missing_from_python2,
            }
            return report

        for key in sorted(py2.files):
            left, right = py2[key], py3[key]
            quantity = key.split("__", 1)[1]
            exact = quantity in DISCRETE
            equal = (np.array_equal(left, right) if exact else
                     np.allclose(left, right, rtol=RTOL, atol=ATOL))
            max_abs = float(np.max(np.abs(left - right))) if left.size else 0.0
            item = {"equal": bool(equal), "exact": exact,
                    "shape": list(left.shape), "max_abs_error": max_abs}
            if not equal:
                indices = np.argwhere(~np.isclose(left, right,
                                                  rtol=RTOL, atol=ATOL))
                item["first_index"] = indices[0].tolist() if indices.size else None
                report["passed"] = False
                if report["first_divergence"] is None:
                    report["first_divergence"] = {"array": key, **item}
            report["arrays"][key] = item
    values = list(report["arrays"].values())
    report["summary"] = {
        "arrays_compared": len(values),
        "exact_arrays": sum(item["exact"] for item in values),
        "tolerance_arrays": sum(not item["exact"] for item in values),
        "maximum_absolute_error": max(
            (item["max_abs_error"] for item in values), default=0.0
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("python2")
    parser.add_argument("python3")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = compare_archives(args.python2, args.python3)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "arrays_compared": len(report["arrays"]), "first_divergence": report["first_divergence"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
