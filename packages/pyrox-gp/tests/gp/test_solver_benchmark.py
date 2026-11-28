"""Smoke tests of ``scripts/bench_gp_solvers.py`` at N = 50 (pyrox#562).

The script writes the table that ``docs/notebooks/gp_solver_choices.ipynb``
loads; these runs check that it still runs against the current API and
that its rows mean what the page says they mean.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "bench_gp_solvers.py"
if not SCRIPT.is_file():
    # A built distribution ships tests/ without the workspace root.
    pytest.skip("scripts/bench_gp_solvers.py is not present", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("bench_gp_solvers", SCRIPT)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_dense_and_matfree_rows_at_n50() -> None:
    problem = bench.make_problem("gaussian", "smooth", 50)
    dense, reference = bench.measure(problem, "dense", keys=2, reference=None)
    row, _ = bench.measure(problem, "matfree", keys=2, reference=reference)
    assert set(dense) == set(bench.COLUMNS) == set(row)
    assert dense["logz_rms"] == 0.0 and dense["grad_rel_rms_max"] == 0.0
    assert 0.0 < float(dense["d_eff"]) < 50.0
    assert row["n_keys"] == 2 and row["passes_evidence"] > 0
    assert row["t_evidence_s"] > 0.0 and row["t_grad_s"] > 0.0
    # rank 50 = N: the Nyström preconditioner is exact, so the estimate
    # agrees with the dense evidence to rounding (1e-14 measured)
    assert row["logz_rel_rms"] < 1e-8
    assert row["grad_rel_rms_max"] < 1e-8
    assert row["pred_mean_rel_err"] < 1e-6


@pytest.mark.slow
def test_full_plan_at_n50_writes_the_table(tmp_path: Path) -> None:
    out = tmp_path / "bench.csv"
    rows = bench.main(out=out, sizes=(50,), keys=2, sweep_n=50)
    with out.open() as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(rows) > 0
    assert tuple(written[0]) == bench.COLUMNS
    assert len({row["id"] for row in written}) == len(written)
    errors = np.array([float(row["logz_rel_rms"]) for row in written])
    assert np.all(np.isfinite(errors))
