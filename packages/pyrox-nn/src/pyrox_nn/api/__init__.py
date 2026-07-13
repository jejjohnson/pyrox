"""User-facing sklearn-style estimator facades.

* `EstimatorBase` — minimal immutable facade. Subclasses declare
  ``feature_cols`` / ``target_col`` plus model-specific hyperparameters;
  override ``fit`` to return a `FittedEstimator`.
* `FittedEstimator` — output of ``fit``. Holds the fitted
  parameters and implements ``predict``.
* `BNFEstimator` family — concrete BNF estimators
  (``BNFEstimator``, ``BNFEstimatorMLE``, ``BNFEstimatorVI``) +
  `FittedBNF`.
"""

from pyrox_nn.api._bnf import (
    BNFEstimator,
    BNFEstimatorMLE,
    BNFEstimatorVI,
    FittedBNF,
)
from pyrox_nn.api._estimator import EstimatorBase, FittedEstimator


__all__ = [
    "BNFEstimator",
    "BNFEstimatorMLE",
    "BNFEstimatorVI",
    "EstimatorBase",
    "FittedBNF",
    "FittedEstimator",
]
