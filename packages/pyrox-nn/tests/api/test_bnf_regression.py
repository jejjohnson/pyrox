"""`BNFEstimator` against predictions stored before it delegated to
`NeuralFieldEstimator` (#317).

The reference values were produced by the pre-refactor `BNFEstimator` (main at
c3c1827) on this exact configuration, under the suite's x64 mode, and printed
to 8 decimals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyrox_nn.api import BNFEstimator


def _df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    t = np.linspace(0, 10, 30)
    return pd.DataFrame(
        {
            "t": t,
            "z": np.sin(t) + 0.1 * rng.normal(size=30),
            "s": 0.05 + 0.1 * rng.uniform(size=30),
        }
    )


_STORED = {
    "map_fixed": (
        {},
        [-0.1649524, 0.96987208, 0.2378434, -0.14710121, 0.10640846],
        [-0.41498082, -0.03167122, -0.01835243, -0.30052894, -0.13115495],
        [0.04416229, 1.94358503, 0.4339174, 0.02025418, 0.35054783],
    ),
    "map_learned_sigma": (
        {"noise": "learned", "sigma_col": "s"},
        [-0.14631901, 0.92488957, 0.16781028, 0.12714931, 0.11956187],
        [-0.49926654, -0.12218476, -0.15639351, -0.20895083, -0.22628429],
        [0.13158881, 1.91554207, 0.41617859, 0.46024107, 0.47774356],
    ),
    "vi_fixed": (
        {"inference_kind": "vi"},
        [-0.03126603, 0.08032936, -0.0842706, -0.11454131, -0.183741],
        [-0.44183624, -0.44501044, -0.46274468, -0.48496929, -0.48696998],
        [0.49237079, 0.76688292, 0.41698527, 0.38798911, 0.22361119],
    ),
}


@pytest.mark.parametrize("case", sorted(_STORED))
def test_bnf_estimator_matches_stored_predictions(case):
    kwargs, mean_ref, q10_ref, q90_ref = _STORED[case]
    df = _df()
    est = BNFEstimator(
        feature_cols=("t",),
        target_col="z",
        width=8,
        depth=2,
        fourier_degrees=(2,),
        ensemble_size=2,
        num_epochs=25,
        **kwargs,
    )
    mean, (q10, q90) = est.fit(df, seed=0).predict(df.iloc[::6], quantiles=(0.1, 0.9))
    # The refactor reproduces the old computation op for op (it matched to
    # every printed digit locally); 1e-6 covers the 8-decimal rounding of
    # the stored values and float64 differences across CPUs over 25 steps.
    np.testing.assert_allclose(mean, mean_ref, atol=1e-6)
    np.testing.assert_allclose(q10, q10_ref, atol=1e-6)
    np.testing.assert_allclose(q90, q90_ref, atol=1e-6)
