"""User-facing sklearn-style estimator facades.

* `EstimatorBase` — minimal immutable facade. Subclasses declare
  ``feature_cols`` / ``target_col`` plus model-specific hyperparameters;
  override ``fit`` to return a `FittedEstimator`.
* `FittedEstimator` — output of ``fit``. Holds the fitted
  parameters and implements ``predict``.
* `BNFEstimator` family — concrete BNF estimators
  (``BNFEstimator``, ``BNFEstimatorMLE``, ``BNFEstimatorVI``) +
  `FittedBNF`.
* `NeuralFieldEstimator` — fits any Bayesian neural field on arrays or an
  xarray Dataset; returns a `FittedNeuralField`. The BNF family is this
  estimator on a `BayesianNeuralField` with pandas preprocessing.
"""

from pyrox_nn.api._bnf import (
    BNFEstimator,
    BNFEstimatorMLE,
    BNFEstimatorVI,
    FittedBNF,
)
from pyrox_nn.api._estimator import EstimatorBase, FittedEstimator
from pyrox_nn.api._field import FittedNeuralField, NeuralFieldEstimator


__all__ = [
    "BNFEstimator",
    "BNFEstimatorMLE",
    "BNFEstimatorVI",
    "EstimatorBase",
    "FittedBNF",
    "FittedEstimator",
    "FittedNeuralField",
    "NeuralFieldEstimator",
]
