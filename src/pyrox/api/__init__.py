"""User-facing sklearn-style estimator facades.

* :class:`EstimatorBase` — minimal immutable facade. Subclasses declare
  ``feature_cols`` / ``target_col`` plus model-specific hyperparameters;
  override ``fit`` to return a :class:`FittedEstimator`.
* :class:`FittedEstimator` — output of ``fit``. Holds the fitted
  parameters and implements ``predict``.
* :class:`BNFEstimator` family — concrete BNF estimators
  (``BNFEstimator``, ``BNFEstimatorMLE``, ``BNFEstimatorVI``) +
  :class:`FittedBNF`.
"""

from pyrox.api._bnf import (
    BNFEstimator,
    BNFEstimatorMLE,
    BNFEstimatorVI,
    FittedBNF,
)
from pyrox.api._estimator import EstimatorBase, FittedEstimator


__all__ = [
    "BNFEstimator",
    "BNFEstimatorMLE",
    "BNFEstimatorVI",
    "EstimatorBase",
    "FittedBNF",
    "FittedEstimator",
]
